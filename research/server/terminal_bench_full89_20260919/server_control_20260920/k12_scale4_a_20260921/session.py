"""Pure token/session boundaries, shared by CPU checks and the GPU service."""
import hashlib,json
def digest(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
def request_seed(plan,task,step):return (plan['seed']+int(hashlib.sha256((task+':'+str(step)).encode()).hexdigest()[:8],16))%2**31
def native_ids(tok,messages,generation=True):
    return tok.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=generation,
        enable_thinking=True,reasoning_effort='xhigh',preserve_thinking=True)
class Session:
    def __init__(self,task):self.task=task;self.step=0;self.static=None;self.previous_messages=None;self.bank={};self.hashes={}
    def split(self,tok,messages,step):
        assert step==self.step,'Out-of-order or duplicate task step'
        assert messages and messages[-1]['role']=='user'
        full=native_ids(tok,messages)
        if self.static is None:
            assert step==0
            self.initial_messages=list(messages);self.static=native_ids(tok,messages,False)
            self.initial_question='\n'.join(str(m['content']) for m in messages)
        else:
            assert len(messages)==len(self.previous_messages)+2 and messages[:-2]==self.previous_messages,'Task history changed'
            assert messages[-2]['role']=='assistant'
        assert full[:len(self.static)]==self.static,'Native initial prefix changed'
        older=native_ids(tok,messages[:-2],False) if step else self.static
        assert older[:len(self.static)]==self.static and full[:len(older)]==older,'Native chat prefix mismatch'
        archive=older[len(self.static):];query=full[len(older):]
        static_chunks=[[self.static[0]]]+[self.static[i:i+512] for i in range(1,len(self.static),512)]
        chunks=[archive[i:i+512] for i in range(0,len(archive),512)]
        assert sum(static_chunks,[])+sum(chunks,[])+query==full
        self.previous_messages=list(messages);self.step+=1
        return full,static_chunks,chunks,query
