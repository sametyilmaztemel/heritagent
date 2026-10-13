import sys,re,html
from html.parser import HTMLParser
class P(HTMLParser):
    def __init__(s):
        super().__init__(); s.out=[]; s.skip=0
    def handle_starttag(s,t,a):
        a=dict(a)
        if t in('script','style'): s.skip+=1
        if t=='math' and a.get('alttext'): s.out.append(' $'+a['alttext']+'$ '); s.skip+=1; s.inmath=True
        if t in('p','div','h1','h2','h3','h4','h5','h6','li','tr','br','figcaption','table'): s.out.append('\n')
        if t in('h1','h2','h3','h4','h5','h6'): s.out.append('#'*int(t[1])+' ')
        if t in('td','th'): s.out.append(' | ')
    def handle_endtag(s,t):
        if t in('script','style','math'): s.skip=max(0,s.skip-1)
    def handle_data(s,d):
        if not s.skip: s.out.append(d)
p=P(); p.feed(open(sys.argv[1],encoding='utf-8').read())
t=''.join(p.out); t=re.sub(r'[ \t]+',' ',t); t=re.sub(r'\n\s*\n+','\n\n',t)
open(sys.argv[2],'w').write(t)
