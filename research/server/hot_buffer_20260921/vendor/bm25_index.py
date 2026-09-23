"""Same token-BM25 arithmetic and hop selection, with document statistics prepared offline."""
from collections import Counter
import math
class BM25Index:
    def __init__(self,chunks):
        self.docs=[x.tolist() if hasattr(x,'tolist') else list(x) for x in chunks]
        self.tf=[Counter(x) for x in self.docs];self.lengths=list(map(len,self.docs))
        n=len(self.docs);self.avg=sum(self.lengths)/n if n else 0
        df=Counter(t for c in self.tf for t in c)
        self.idf={t:math.log((n-v+.5)/(v+.5)+1.) for t,v in df.items()}
    def scores(self,query):
        terms=set(int(t) for t in query);scores=[]
        for tf,dl in zip(self.tf,self.lengths):
            s=0.
            for t in terms:
                f=tf.get(t,0)
                if not f:continue
                denom=f+1.5*(1.-.75+.75*dl/self.avg) if self.avg>0 else f+1.5
                s+=self.idf.get(t,0.)*(f*2.5)/denom
            scores.append(s)
        return scores
    def select(self,query,k=12,hop=2):
        selected=[];seen=set();frontier=list(query)
        for _ in range(-(-k//hop)):
            if len(selected)>=k or not frontier:break
            scores=self.scores(frontier)
            cand=[i for i in range(len(self.docs)) if i not in seen and scores[i]>0.]
            cand.sort(key=lambda i:scores[i],reverse=True)
            new=cand[:min(hop,k-len(selected))]
            if not new:break
            selected.extend(new);seen.update(new)
            frontier=[t for i in new for t in self.docs[i]]
        return sorted(selected)
