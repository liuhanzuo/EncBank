"""Focused tests of the new entry boundary; no GPU execution here."""
import ast, json, types, weakref
import torch
from extracted_minmax import quantize_tensor, _pack_unsigned
from packed_reader import split_fixture, encode_entries, materialize
from common import ROOT, dump

def main():
    torch.manual_seed(42)
    for n in (0,1,63,64,65,511,512,513):
        x=torch.randn(1,n,7,dtype=torch.bfloat16)
        for bits in (16,8,4):
            p=quantize_tensor(x,bits=bits);y=p.dequantize()
            assert y.shape==x.shape and y.dtype==x.dtype
            if bits==16:
                assert torch.equal(x,y)
                if n:assert p.data.data_ptr()!=y.data_ptr()
            else:
                assert p.scales.dtype==p.biases.dtype==torch.bfloat16
                assert p.nbytes==((x.numel()+63)//64)*(68 if bits==8 else 36)
            before=p.dequantize().clone();y.fill_(123)
            assert torch.equal(before,p.dequantize())
    assert _pack_unsigned(torch.tensor([1,2,3,4],dtype=torch.uint8),4).tolist()==[33,67]
    constant=torch.full((1,7,3),1.25,dtype=torch.bfloat16)
    assert torch.equal(quantize_tensor(constant,bits=4).dequantize(),constant)
    class Fake:
        device=torch.device('cpu')
        def write(self,ids):return torch.tensor(ids,dtype=torch.bfloat16).reshape(1,-1,1).repeat(1,1,7)
    reader=Fake();source=[list(range(512)),list(range(512)),[1,2,3]]
    entries=encode_entries(reader,source,99,[16,8,4],'test')
    for bit,e in entries.items():
        assert e.inventory()['blocks']==3 and e.inventory()['source_tokens']==1027
        assert e.inventory()['document_lower_KV_bytes']==0
        before=e.digest();one=materialize(e,[0,2],'test')
        assert one.shape==(1,516,7)
        one.fill_(-999);assert before==e.digest()
        assert materialize(e,[],'test').shape==(1,1,7)
        ref=weakref.ref(e.chunks[0].data);e.release();assert ref() is None
        try:e.inventory();raise RuntimeError('release failed')
        except AssertionError:pass
    row=dict(input_ids=list(range(1027)),selected=[1],sink=2,
             segments=[[2],list(range(512,1024)),list(range(1024,1027))],source_tokens=1024)
    s,q,sel=split_fixture(row);assert len(s)==2 and len(q)==3 and sel==[1]
    row['selected']=[];row['segments']=[[2],list(range(1024,1027))]
    assert split_fixture(row)[2]==[]
    dump(ROOT/'cpu_checks.json',dict(passed=True,device='CPU',
        checks=['padding and byte packing','BF16 metadata and H16 exactness','no dequantization alias',
                'all source chunks retained','selected/empty materialization','entry immutability and release',
                'original fixture boundary'],
        gpu_check='100-example formal H16 cell must match old native generated IDs before expansion'))
    print('Focused CPU checks passed.')

if __name__=='__main__':main()
