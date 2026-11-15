"""Asset isolation preserves uncertainty and reuses strict scientific validators."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cloud"))
from test_cloud_repair_finalize import complete
import cloud_repair_finalize as final
import cloud_repair_isolation as isolation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def isolated(complete):
    artifacts, old = complete
    final.finalize(artifacts, old)
    (artifacts / "cleanup-authorization.json").unlink()
    campaign = json.loads((artifacts / "campaign.json").read_text())
    campaign["source_revision"] = "c" * 40
    campaign["superseded_unreceived_gpu_intents"] = []
    for worker in (0, 1):
        log = artifacts / f"original-worker{worker}.log"
        log.write_text("transport timeout with no receipt\n")
        campaign["superseded_unreceived_gpu_intents"].append({
            "intent":f"worker-{worker}", "note":isolation.REPLACEABLE_NOTES[worker],
            "mode":"worker", "worker":worker, "gpus":1,
            "source_revision":isolation.ORIGINAL_SOURCES[worker],
            "original_still_reserves_gpu":True,"user_override":isolation.USER_OVERRIDE,
            "submission_evidence":str(log),"submission_evidence_sha256":isolation.digest(log)})
    cpu={"intent":"inspect-4","note":"repair8b_inspect-4_exact", "mode":"inspect",
         "worker":None,"gpus":0,"source_revision":"d"*40}
    (artifacts / "submission-inspect-4.log").write_text("timeout without ID\n")
    campaign["unresolved_auxiliary_submissions"]=[cpu]
    write(artifacts / "campaign.json", campaign)
    scheduler=json.loads((artifacts / "scheduler-snapshot.json").read_text())
    for row in scheduler['jobs']:
        if row['status']=='Running':
            row['status']='Completed'; row['evidence']=row['evidence'].replace(' Running ',' Completed ')
    raw=Path(scheduler['raw_proof'])
    raw.write_text(raw.read_text().replace(' Running ',' Completed '))
    scheduler['raw_proof_sha256']=isolation.digest(raw)
    write(artifacts / "scheduler-snapshot.json", scheduler)
    unknown=isolation.unknown_ledger(campaign,artifacts)
    history=artifacts/'source-gates/history.json'
    write(history,{'campaign':isolation.CAMPAIGN,'complete_published_history':True,'ordering':'oldest_to_newest','published_revisions':['d'*40,isolation.ORIGINAL_SOURCES[0],isolation.ORIGINAL_SOURCES[1],'c'*40]})
    sources=[]
    for request in unknown['unknown_requests']:
        paths=['cloud_repair_runtime.py'] if request['mode']=='worker' else ['cloud_repair_entry.py','cloud_repair_control.py']
        for name in paths:
            source=artifacts/'source-gates'/request['source_revision']/name
            source.parent.mkdir(parents=True,exist_ok=True)
            source.write_text('# retained exact source blob for independently reviewed fixture\n')
            sources.append({'source_revision':request['source_revision'],'path':'scripts/cloud/'+name,
                'source_path':str(source.relative_to(artifacts)),'sha256':isolation.digest(source),
                'gates':['complete_claim_before_assets_cuda' if request['mode']=='worker' else 'read_only_metadata']})
    source=artifacts/'source-gates/current-runtime.py';source.write_text('# current reviewed closure gates\n')
    sources.append({'source_revision':'c'*40,'path':'scripts/cloud/cloud_repair_runtime.py',
        'source_path':str(source.relative_to(artifacts)),'sha256':isolation.digest(source),
        'gates':['worker_closed_before_assets_cuda','profile_closed_before_assets_cuda']})
    for name in ('cloud_repair_entry.py','cloud_repair_control.py'):
        current_blob=artifacts/'source-gates'/('current-'+name)
        current_blob.write_text('# additional current published CPU metadata gate\n')
        sources.append({'source_revision':'c'*40,'path':'scripts/cloud/'+name,
            'source_path':str(current_blob.relative_to(artifacts)),'sha256':isolation.digest(current_blob),
            'gates':['read_only_metadata']})
    # The old inspect request can clone every later published revision.
    for revision in (isolation.ORIGINAL_SOURCES[0],isolation.ORIGINAL_SOURCES[1]):
        for name in ('cloud_repair_entry.py','cloud_repair_control.py'):
            blob=artifacts/'source-gates'/revision/name
            blob.write_text('# retained intermediate published read-only inspector source\n')
            sources.append({'source_revision':revision,'path':'scripts/cloud/'+name,
                'source_path':str(blob.relative_to(artifacts)),'sha256':isolation.digest(blob),
                'gates':['read_only_metadata']})
    proof={'campaign':isolation.CAMPAIGN,'audit_kind':'cpu_metadata_only_no_model_or_cuda',
        'prepared_by':'root-agent','current_revision':'c'*40,'unknown_requests':unknown['unknown_requests'],'sources':sources,
        'published_history':{'path':str(history.relative_to(artifacts)),'sha256':isolation.digest(history)}}
    review={'proof':proof,'review':{'decision':isolation.SOURCE_REVIEW_DECISION,
        'proof_sha256':isolation.canonical_sha(proof),'reviewer':'independent-test-reviewer','independent':True}}
    review_path=artifacts/'source-gate-review.json';write(review_path,review)
    return artifacts,review_path


def edit(artifacts,filename,change):
    path=artifacts/filename;value=json.loads(path.read_text());change(value);write(path,value)


def test_complete_isolation_proof_keeps_unknown_ledger_and_never_edits_campaign(isolated):
    artifacts,review=isolated
    before=(artifacts/'campaign.json').read_bytes()
    result=isolation.build_proof(artifacts,review)
    allocation=result['allocation_proof']
    assert allocation['all_known_jobs_terminal'] is True and allocation['all_owned_jobs_terminal'] is False
    assert len(allocation['unknown_requests'])==3 and len(result['scientific_verification'])==8
    assert allocation['ledger_sha256']==isolation.canonical_sha(json.loads(before))
    assert allocation['unknown_ledger_sha256']==isolation.canonical_sha(allocation['canonical_unknown_ledger'])
    assert (artifacts/'campaign.json').read_bytes()==before
    assert not (artifacts/'cleanup-authorization.json').exists()
    request=isolation.closure_request(result,{'decision':isolation.CLOSURE_REVIEW_DECISION,
        'proof_sha256':isolation.canonical_sha(result),'review_sha256':'f'*64})
    assert request['proof']==result
    request['proof']['allocation_proof']['unknown_requests'].clear()
    assert result['allocation_proof']['unknown_requests']


@pytest.mark.parametrize('damage',['mainpending','foreigngpu','receipt','observed','conflict','source','cpumode','cpugpus','duplicate','cpucap','knownnote'])
def test_unknown_scope_or_positive_identity_evidence_fails_closed(isolated,damage):
    artifacts,review=isolated
    def change(c):
        gpu=c['superseded_unreceived_gpu_intents'][0];cpu=c['unresolved_auxiliary_submissions'][0]
        if damage=='mainpending':c['ambiguous_submissions']=[deepcopy(cpu)]
        if damage=='foreigngpu':gpu['worker']=2
        if damage=='receipt':gpu['job_id']='lm-mpi-job-positive'
        if damage=='observed':gpu['observed_job_ids']=['lm-mpi-job-positive']
        if damage=='conflict':gpu['recovery_conflict']={'reason':'conflicting evidence'}
        if damage=='source':gpu['source_revision']='e'*40
        if damage=='cpumode':cpu['mode']='prepare'
        if damage=='cpugpus':cpu['gpus']=True
        if damage=='duplicate':c['unresolved_auxiliary_submissions'].append(deepcopy(cpu))
        if damage=='cpucap':c['unresolved_auxiliary_submissions']*=5
        if damage=='knownnote':c['auxiliary_jobs'][0]['note']=cpu['note']
    edit(artifacts,'campaign.json',change)
    with pytest.raises(ValueError):isolation.build_proof(artifacts,review)


@pytest.mark.parametrize('damage',['knownactive','inspectactive','partialhistory','sharedclaimed','export','sourcereview','sourceblob','historyblob','missinggate','requestbinding','currentrevision'])
def test_missing_bound_proof_or_known_active_allocation_refuses_isolation(isolated,damage):
    artifacts,review=isolated
    if damage in {'knownactive','inspectactive'}:
        scheduler=json.loads((artifacts/'scheduler-snapshot.json').read_text())
        target=scheduler['jobs'][0] if damage=='knownactive' else scheduler['jobs'][-2]
        if damage=='inspectactive':
            current=json.loads((artifacts/'inspection-latest.json').read_text())['collector_source_job_id']
            target=next(row for row in scheduler['jobs'] if row['job_id']==current)
        old=target['evidence'];target['status']='Running';target['evidence']=old.replace(' Completed ',' Running ')
        raw=Path(scheduler['raw_proof']);raw.write_text(raw.read_text().replace(old,target['evidence']))
        scheduler['raw_proof_sha256']=isolation.digest(raw);write(artifacts/'scheduler-snapshot.json',scheduler)
    if damage=='partialhistory':
        history=next((artifacts/'raw-wandb').glob('*/history.jsonl'));history.write_text(history.read_text().splitlines()[0]+'\n')
    if damage=='sharedclaimed':edit(artifacts,'inspection-latest.json',lambda s:s['tasks'][0].update(phase='claimed'))
    if damage=='export':(artifacts/'final/paired-results.csv').write_text('changed')
    envelope=json.loads(review.read_text())
    if damage=='sourcereview':envelope['review']['proof_sha256']='0'*64
    if damage=='sourceblob':(artifacts/envelope['proof']['sources'][0]['source_path']).write_text('changed')
    if damage=='historyblob':(artifacts/envelope['proof']['published_history']['path']).write_text('changed')
    if damage=='missinggate':
        next(row for row in envelope['proof']['sources'] if row['source_revision']=='c'*40 and row['path'].endswith('cloud_repair_runtime.py'))['gates']=[]
    if damage=='requestbinding':envelope['proof']['unknown_requests'][0]['source_revision']='e'*40
    if damage=='currentrevision':envelope['proof']['current_revision']='e'*40
    if damage in {'missinggate','requestbinding','currentrevision'}:
        envelope['review']['proof_sha256']=isolation.canonical_sha(envelope['proof'])
    write(review,envelope)
    with pytest.raises(ValueError):isolation.build_proof(artifacts,review)


def test_closure_review_is_required_and_digest_bound(isolated):
    proof=isolation.build_proof(*isolated)
    with pytest.raises(ValueError):isolation.closure_request(proof,{'decision':isolation.CLOSURE_REVIEW_DECISION,
        'proof_sha256':'0'*64,'review_sha256':'f'*64})


def test_real_isolation_proof_matches_remote_closure_validator(isolated,monkeypatch):
    import cloud_repair_closure as closure
    artifacts,review=isolated
    bundle=artifacts/'fixture-bundle.sha256';bundle.write_text('a'*64)
    monkeypatch.setattr(closure,'BUNDLE_SHA',bundle)
    proof=isolation.build_proof(artifacts,review)
    request=isolation.closure_request(proof,{'decision':isolation.CLOSURE_REVIEW_DECISION,
        'proof_sha256':isolation.canonical_sha(proof),'review_sha256':'f'*64})
    assert closure.reviewed_proof(request)==proof


def test_retained_unknown_metadata_changes_fingerprint_and_invalidates_prior_review(isolated):
    artifacts,review=isolated
    before=isolation.build_proof(artifacts,review)
    approval={'decision':isolation.CLOSURE_REVIEW_DECISION,
        'proof_sha256':isolation.canonical_sha(before),'review_sha256':'f'*64}
    edit(artifacts,'campaign.json',lambda c:c['superseded_unreceived_gpu_intents'][0].update(retained_audit_note='new evidence retained'))
    after=isolation.build_proof(artifacts,review)
    assert after['allocation_proof']['unknown_ledger_sha256']!=before['allocation_proof']['unknown_ledger_sha256']
    with pytest.raises(ValueError):isolation.closure_request(after,approval)


def test_digest_review_cannot_replace_structured_scientific_proof_with_boolean():
    proof={'campaign':isolation.CAMPAIGN,'all_results_complete':True}
    with pytest.raises(ValueError,match='structured eight-result'):
        isolation.closure_request(proof,{'decision':isolation.CLOSURE_REVIEW_DECISION,
            'proof_sha256':isolation.canonical_sha(proof),'review_sha256':'f'*64})


@pytest.mark.parametrize('damage',['missingworkerblob','missinginspectblob','unreviewedintermediate','currentnotlast','unordered'])
def test_every_possible_late_clone_revision_requires_relevant_reviewed_source(isolated,damage):
    artifacts,review_path=isolated
    envelope=json.loads(review_path.read_text())
    proof=envelope['proof']
    history_path=artifacts/proof['published_history']['path']
    history=json.loads(history_path.read_text())
    if damage=='missingworkerblob':
        proof['sources']=[row for row in proof['sources'] if not (row['source_revision']==isolation.ORIGINAL_SOURCES[1] and row['path'].endswith('cloud_repair_runtime.py'))]
    if damage=='missinginspectblob':
        proof['sources']=[row for row in proof['sources'] if not (row['source_revision']==isolation.ORIGINAL_SOURCES[0] and row['path'].endswith('cloud_repair_control.py'))]
    if damage=='unreviewedintermediate':history['published_revisions'].insert(-1,'e'*40)
    if damage=='currentnotlast':history['published_revisions'].append('e'*40)
    if damage=='unordered':history.pop('ordering')
    write(history_path,history)
    proof['published_history']['sha256']=isolation.digest(history_path)
    envelope['review']['proof_sha256']=isolation.canonical_sha(proof)
    write(review_path,envelope)
    with pytest.raises(ValueError):isolation.build_proof(artifacts,review_path)
