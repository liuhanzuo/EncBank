"""Single natural-generation Read diagnostics, not fixed-work infra."""
import time
class QualityTiming:
    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        torch.cuda.synchronize(device)
        self.start = time.perf_counter()
        self.first = self.last = None
        self.count = 0
    def sampled(self):
        now = time.perf_counter()  # Follows argmax.item(), which waits for the sampled token.
        if self.first is None: self.first = now
        self.last = now
        self.count += 1
    def finish(self, generated_tokens):
        self.torch.cuda.synchronize(self.device)
        end = time.perf_counter()
        assert self.count == generated_tokens and self.count >= 1
        decode = self.last - self.first if self.count > 1 else None
        return {'generated_tokens':self.count, 'ttft_seconds':self.first-self.start,
                'decode_tokens':max(0,self.count-1), 'decode_seconds':decode,
                'decode_tokens_per_second':(self.count-1)/decode if decode is not None and decode > 0 else None,
                'read_total_seconds':end-self.start,
                'scope':'Single variable-length natural quality Read; TTFT includes Read preparation and full query; decode excludes first token and EOS decision; total includes EOS decision and request cleanup. No Write/model load included; not fixed-work or cross-device infrastructure.'}
