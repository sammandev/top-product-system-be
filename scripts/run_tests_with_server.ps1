$p = Start-Process -FilePath 'd:\Projects\AST_Parser\backend_fastapi\.venv\Scripts\python.exe' -ArgumentList '-m','uvicorn','app.main:app','--port','8001' -WorkingDirectory 'd:\Projects\AST_Parser\backend_fastapi' -PassThru
Start-Sleep -Seconds 3
$env:ASTPARSER_TEST_BASE='http://127.0.0.1:8001'
& 'd:\Projects\AST_Parser\backend_fastapi\.venv\Scripts\python.exe' -m pytest backend_fastapi/tests -q
Stop-Process -Id $p.Id -Force
