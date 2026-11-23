#!/usr/bin/env python
# coding: utf-8

# # DSA 8401 TGEE — FastAPI scoring service
# 
# This notebook loads `artifacts/tgee_model_bundle.joblib` created by `01_TGEE_End_to_End.ipynb` and exposes single-record and batch scoring endpoints.
# 
# **Input contract:** the API accepts a *prepared feature record* produced by the same point-in-time feature-engineering process as the training notebook. It does not accept a raw OCDS release and independently reconstruct graph history; clients must supply the temporal and graph-history feature fields. The API returns percentile scores and review bands, **not fraud probabilities**.

# In[1]:


import os, json
from pathlib import Path
from typing import Optional, Any
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.testclient import TestClient

ROOT=Path.cwd()
if not (ROOT/'artifacts').exists() and (ROOT.parent/'artifacts').exists(): ROOT=ROOT.parent
BUNDLE_PATH=ROOT/'artifacts'/'tgee_model_bundle.joblib'
if not BUNDLE_PATH.exists():
    raise FileNotFoundError('Model bundle not found. Run notebooks/01_TGEE_End_to_End.ipynb first.')
bundle=joblib.load(BUNDLE_PATH)

app=FastAPI(
    title='Kenya PPIP Procurement Integrity TGEE API',
    version=bundle.get('model_version','1.0.0'),
    description='Human-review prioritisation scores. Not a fraud determination or probability.'
)

class PreparedRecord(BaseModel):
    record_id: Optional[str] = None
    ocid: Optional[str] = None
    context: dict[str, Any] = Field(default_factory=dict)
    tabular: dict[str, Any]
    temporal: dict[str, Any]
    graph: dict[str, Any]
    text: Optional[str] = ''
    hypergraph: Optional[dict[str, Any]] = None

class BatchRequest(BaseModel):
    records: list[PreparedRecord] = Field(min_length=1, max_length=1000)


def _percentile(reference, value):
    ref=np.asarray(reference,dtype=float)
    ref=ref[np.isfinite(ref)]
    if len(ref)==0 or value is None or not np.isfinite(float(value)):
        return 0.5
    return float(np.searchsorted(np.sort(ref),float(value),side='right')/len(ref))


def score_one(payload: PreparedRecord):
    # Construct a one-row DataFrame in the exact training feature order.
    tab_cols=bundle['tabular_feature_cols']
    tab_row={k:payload.tabular.get(k,None) for k in tab_cols}
    tab_frame=pd.DataFrame([tab_row],columns=tab_cols)
    tab_raw=float(-bundle['tabular_model'].score_samples(tab_frame)[0])
    tab_score=_percentile(bundle['tabular_reference'],tab_raw)

    temporal_cols=bundle['temporal_feature_cols']
    temporal_row={k:payload.temporal.get(k,None) for k in temporal_cols}
    temporal_frame=pd.DataFrame([temporal_row],columns=temporal_cols)
    temp_raw=float(-bundle['temporal_model'].score_samples(temporal_frame)[0])
    temp_score=_percentile(bundle['temporal_reference'],temp_raw)

    graph_cols=bundle['graph_feature_cols']
    graph_row={k:payload.graph.get(k,None) for k in graph_cols}
    graph_frame=pd.DataFrame([graph_row],columns=graph_cols)
    graph_raw=float(-bundle['graph_model'].score_samples(graph_frame)[0])
    graph_score=_percentile(bundle['graph_reference'],graph_raw)

    components={'tabular':tab_score,'temporal':temp_score,'graph':graph_score}
    if bundle.get('text_active') and payload.text:
        X=bundle['text_vectorizer'].transform([payload.text])
        X=bundle['text_svd'].transform(X)
        raw=float(-bundle['text_detector'].score_samples(X)[0])
        components['text']=_percentile(bundle['text_reference'],raw)
    if bundle.get('hypergraph_active') and payload.hypergraph is not None:
        hcols=bundle['hypergraph_feature_cols']
        hrow={k:payload.hypergraph.get(k,None) for k in hcols}
        hframe=pd.DataFrame([hrow],columns=hcols)
        raw=float(-bundle['hypergraph_model'].score_samples(hframe)[0])
        components['hypergraph']=_percentile(bundle['hypergraph_reference'],raw)

    weights=bundle['weights']
    available={k:v for k,v in weights.items() if k in components}
    weight_sum=sum(available.values()) or 1.0
    score=float(sum(components[k]*available[k] for k in available)/weight_sum)
    if score >= bundle['high_threshold']:
        band='high_priority'
    elif score >= bundle['review_threshold']:
        band='review'
    else:
        band='routine'
    return {
        'record_id':payload.record_id,'ocid':payload.ocid,
        'tgee_priority_score':round(score,6),'review_band':band,
        'component_scores':{k:round(v,6) for k,v in components.items()},
        'score_semantics':'Relative review-priority percentile; not a fraud probability.',
        'context':payload.context
    }

@app.get('/health')
def health():
    return {'status':'ok','model_version':bundle.get('model_version'),
            'data_mode':bundle.get('data_mode'),'text_active':bundle.get('text_active'),
            'hypergraph_active':bundle.get('hypergraph_active')}

@app.get('/model-info')
def model_info():
    return {
        'project_name':bundle.get('project_name'),
        'model_version':bundle.get('model_version'),
        'active_branches':list(bundle.get('weights',{}).keys()),
        'weights':bundle.get('weights'),
        'review_threshold':bundle.get('review_threshold'),
        'high_threshold':bundle.get('high_threshold'),
        'hypergraph_coverage':bundle.get('hypergraph_coverage'),
        'score_semantics':bundle.get('score_semantics')
    }

@app.post('/score')
def score_endpoint(record: PreparedRecord):
    try:
        return score_one(record)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f'Unable to score prepared record: {exc}')

@app.post('/score/batch')
def score_batch(payload: BatchRequest):
    try:
        return {'count':len(payload.records),'results':[score_one(r) for r in payload.records]}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f'Unable to score batch: {exc}')


# ## API smoke test

# In[2]:


# Build an example request from the latest processed synthetic/empirical feature table.
scored_path=ROOT/'data'/'processed'/'api_prepared_features.csv'
if not scored_path.exists():
    raise FileNotFoundError('Run the end-to-end TGEE notebook before running this API notebook.')
scored=pd.read_csv(scored_path)
row=scored.iloc[0]

# The processed score file contains key model inputs; if a categorical value is missing, pass it as null.
def json_safe(v):
    if v is None or pd.isna(v):
        return None
    if isinstance(v,np.generic):
        return v.item()
    return v

request_payload={
    'record_id':str(row.get('release_id','demo-record')),
    'ocid':None if pd.isna(row.get('ocid')) else str(row.get('ocid')),
    'context':{'buyer_name':None if pd.isna(row.get('buyer_name')) else str(row.get('buyer_name'))},
    'tabular':{k:json_safe(row.get(k)) for k in bundle['tabular_feature_cols'] if k in row.index},
    'temporal':{k:json_safe(row.get(k)) for k in bundle['temporal_feature_cols'] if k in row.index},
    'graph':{k:json_safe(row.get(k)) for k in bundle['graph_feature_cols'] if k in row.index},
    'text':'' if pd.isna(row.get('combined_text')) else str(row.get('combined_text')),
    'hypergraph':({k:json_safe(row.get(k)) for k in bundle['hypergraph_feature_cols'] if k in row.index} if bundle.get('hypergraph_active') else None)
}
# Uses the same engineered feature columns and text prepared by notebook 1.
client=TestClient(app)
print('GET /health:',client.get('/health').json())
print('GET /model-info:',client.get('/model-info').json())
response=client.post('/score',json=request_payload)
print('POST /score status:',response.status_code)
print(response.json())
assert response.status_code==200


# In[3]:


# Start a persistent Uvicorn server when invoked by run_api.sh or Docker.
if os.getenv('START_API','0') == '1':
    import uvicorn
    uvicorn.run(app,host=os.getenv('API_HOST','0.0.0.0'),port=int(os.getenv('API_PORT','8000')))
else:
    print('API app is ready. For a persistent server run: bash run_api.sh')


# In[ ]:




