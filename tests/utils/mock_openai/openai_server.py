from fastapi import FastAPI
from uvicorn import run
app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request:dict):
    request_str = str(request)
    is_affiliation = "evaluates their institutional prestige" in request_str
    is_notable_author = "evaluates the academic reputation" in request_str
    if is_affiliation:
        content = '{"affiliations": ["The Pennsylvania State University"], "first_author_institution": "The Pennsylvania State University", "is_qs_top100": false}'
    elif is_notable_author:
        content = '{"corresponding_author": "Abhronil Sengupta", "is_notable": false}'
    else:
        content = 'Hello! How can I assist you today?'
    return {'id': 'chatcmpl-CkUpDqPLWNJE4SZCoPsUbvf3RudrU',
 'created': 1765197615,
 'model': 'gpt-4o-mini-2024-07-18',
 'object': 'chat.completion',
 'system_fingerprint': 'fp_efad92c60b',
 'choices': [{'finish_reason': 'stop',
   'index': 0,
   'message': {'content': content,
    'role': 'assistant',
    'annotations': []},
   'provider_specific_fields': {'content_filter_results': {'hate': {'filtered': False,
      'severity': 'safe'},
     'protected_material_code': {'filtered': False, 'detected': False},
     'protected_material_text': {'filtered': False, 'detected': False},
     'self_harm': {'filtered': False, 'severity': 'safe'},
     'sexual': {'filtered': False, 'severity': 'safe'},
     'violence': {'filtered': False, 'severity': 'safe'}}}}],
 'usage': {'completion_tokens': 10,
  'prompt_tokens': 9,
  'total_tokens': 19,
  'completion_tokens_details': {'accepted_prediction_tokens': 0,
   'audio_tokens': 0,
   'reasoning_tokens': 0,
   'rejected_prediction_tokens': 0},
  'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}},
 'prompt_filter_results': [{'prompt_index': 0,
   'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'},
    'jailbreak': {'filtered': False, 'detected': False},
    'self_harm': {'filtered': False, 'severity': 'safe'},
    'sexual': {'filtered': False, 'severity': 'safe'},
    'violence': {'filtered': False, 'severity': 'safe'}}}]}

@app.post("/v1/embeddings")
async def embeddings(request:dict):
    return {'model': 'text-embedding-3-large',
 'data': [{'embedding': [0.1, 0.2, 0.3], 'index': 0, 'object': 'embedding'}] * len(request['input']),
 'object': 'list',
 'usage': {'completion_tokens': 0,
  'prompt_tokens': 2,
  'total_tokens': 2,
  'completion_tokens_details': None,
  'prompt_tokens_details': None}}

if __name__ == "__main__":
    run(app, host="0.0.0.0", port=30000)