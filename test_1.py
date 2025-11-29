import httpx, time
resp = httpx.post("http://127.0.0.1:8000/quiz", json={
  "email":"23f2003519@ds.study.iitm.ac.in",
  "secret":"this_is_pinga",
  "url":"https://tds-llm-analysis.s-anand.net/demo"
})
print(resp.status_code, resp.json())
# {
#   "email": "your email",
#   "secret": "your secret",
#   "url": "https://tds-llm-analysis.s-anand.net/demo"
# }
