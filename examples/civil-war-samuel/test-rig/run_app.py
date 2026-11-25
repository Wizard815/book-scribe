"""Start the real app with a long model timeout (the stand-in model is an agent that takes a minute or two per answer)."""
import sys
sys.path.insert(0, "/home/user/book-scribe/agent")
import clients
clients.DEFAULT_TIMEOUT.update({"chat": 1500.0, "image": 120.0})
import uvicorn
uvicorn.run("main:create_app", factory=True, host="127.0.0.1", port=8077, log_level="warning")
