"""Bounded, validated range reads of public fixed-revision archives."""
import io
import urllib.request


class RangeReader(io.RawIOBase):
    def __init__(self,url,size):
        self.url=url;self.size=size;self.position=0;self.bytes_received=0;self.requests=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0:raise ValueError('Negative offset')
        return self.position
    def read(self,size=-1):
        if size<0:size=self.size-self.position
        size=min(size,self.size-self.position)
        if size==0:return b''
        if size>64*1024*1024:raise ValueError(f'Refusing >64MiB read: {size}')
        start=self.position;end=start+size-1
        # Separate cache keys prevent an intermediary serving a previous range.
        url=self.url+('&' if '?' in self.url else '?')+f'range_start={start}&range_end={end}'
        req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
        for retry in range(3):
            try:
                with urllib.request.urlopen(req,timeout=60) as r:
                    if r.status!=206:raise IOError('Range ignored; whole archive download prevented')
                    if r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':raise IOError('Content-Range mismatch')
                    data=r.read(size+1)
                if len(data)!=size:raise IOError('Truncated range')
                break
            except Exception:
                if retry==2:raise
        self.position+=size;self.bytes_received+=size;self.requests+=1
        return data
