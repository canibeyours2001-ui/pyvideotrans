"""Single-host durable queue for Lightning AI or a Linux GPU server."""
import json,os,sqlite3,time,threading
from pathlib import Path
from contextlib import asynccontextmanager
from backend.service import make_app,execute
ROOT=Path(os.getenv('STUDIO_DATA','./studio-data')).resolve();ROOT.mkdir(exist_ok=True,parents=True)
class Store:
    def __init__(self):
        with self.db() as db:db.execute('CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    def db(self):
        db=sqlite3.connect(ROOT/'jobs.sqlite',timeout=30);db.execute('PRAGMA journal_mode=WAL');return db
    def get(self,key):
        with self.db() as db:r=db.execute('SELECT value FROM records WHERE key=?',(key,)).fetchone()
        return json.loads(r[0]) if r else None
    def set(self,key,value):
        with self.db() as db:db.execute('INSERT INTO records VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(key,json.dumps(value)))
    def delete(self,key):
        with self.db() as db:db.execute('DELETE FROM records WHERE key=?',(key,))
    def items(self):
        with self.db() as db:return [(k,json.loads(v)) for k,v in db.execute('SELECT key,value FROM records')]
    def lock(self,key):
        with self.db() as db:return db.execute('INSERT OR IGNORE INTO records VALUES (?,?)',('lock:'+key,json.dumps({'expires':time.time()+3600}))).rowcount==1
    def unlock(self,key):self.delete('lock:'+key)
store=Store()
def dispatch(job,action):store.set('queue:'+job,{'job':job,'action':action})
api=make_app(store,ROOT,dispatch)
stop=threading.Event()
def worker():
    while not stop.wait(1):
        for key,data in store.items():
            if key.startswith('queue:'):
                execute(store,ROOT,data['job'],data['action']);store.delete(key)
@asynccontextmanager
async def lifespan(app):
    # This deployment must run exactly one Uvicorn process.
    thread=threading.Thread(target=worker,daemon=True);thread.start()
    yield
    stop.set();thread.join(timeout=5)
api.router.lifespan_context=lifespan
