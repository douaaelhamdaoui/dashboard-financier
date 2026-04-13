import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def relu(z): return np.maximum(0,z)
def softmax(z):
    e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
def loss(p,y): return -np.mean(np.sum(y*np.log(p+1e-12),1))
def acc(p,y): return np.mean(p.argmax(1)==y.argmax(1))

class MLP:
    def __init__(self):
        self.W=[np.random.randn(784,256)*np.sqrt(2/784),np.random.randn(256,128)*np.sqrt(2/256),np.random.randn(128,10)*np.sqrt(2/128)]
        self.b=[np.zeros((1,s)) for s in [256,128,10]]
    def fwd(self,X):
        self.a=[X];self.z=[]
        for i,(w,b) in enumerate(zip(self.W,self.b)):
            z=self.a[-1]@w+b;self.z.append(z)
            self.a.append(softmax(z) if i==2 else relu(z))
        return self.a[-1]
    def bwd(self,y):
        m=y.shape[0];dz=self.a[-1]-y;gW=[];gb=[]
        for i in range(2,-1,-1):
            gW.insert(0,self.a[i].T@dz/m);gb.insert(0,dz.mean(0,keepdims=True))
            if i>0: dz=(dz@self.W[i].T)*(self.z[i-1]>0)
        return gW,gb

def run(opt_fn,epochs=20,bs=128):
    np.random.seed(42);m=MLP();s={};h={'tl':[],'ta':[],'vl':[],'va':[]}
    for ep in range(epochs):
        idx=np.random.permutation(4000);tl=0
        for i in range(0,4000,bs):
            b=idx[i:i+bs];p=m.fwd(Xt[b]);tl+=loss(p,yt[b])*len(b)
            gW,gb=m.bwd(yt[b]);opt_fn(m,gW,gb,s)
        tp=m.fwd(Xt);vp=m.fwd(Xv)
        h['tl'].append(tl/4000);h['ta'].append(acc(tp,yt))
        h['vl'].append(loss(vp,yv));h['va'].append(acc(vp,yv))
    return h

def sgd(m,gW,gb,s,lr=0.01,mu=0.9):
    s.setdefault('v',[(np.zeros_like(w),np.zeros_like(b)) for w,b in zip(m.W,m.b)])
    for i in range(3):
        s['v'][i]=(mu*s['v'][i][0]-lr*gW[i],mu*s['v'][i][1]-lr*gb[i])
        m.W[i]+=s['v'][i][0];m.b[i]+=s['v'][i][1]

def adam(m,gW,gb,s,lr=0.001,b1=0.9,b2=0.999,eps=1e-8):
    s.setdefault('t',0);s['t']+=1;t=s['t']
    s.setdefault('m',[(np.zeros_like(w),np.zeros_like(b)) for w,b in zip(m.W,m.b)])
    s.setdefault('v',[(np.zeros_like(w),np.zeros_like(b)) for w,b in zip(m.W,m.b)])
    for i in range(3):
        s['m'][i]=(b1*s['m'][i][0]+(1-b1)*gW[i],b1*s['m'][i][1]+(1-b1)*gb[i])
        s['v'][i]=(b2*s['v'][i][0]+(1-b2)*gW[i]**2,b2*s['v'][i][1]+(1-b2)*gb[i]**2)
        mh=(s['m'][i][0]/(1-b1**t),s['m'][i][1]/(1-b1**t))
        vh=(s['v'][i][0]/(1-b2**t),s['v'][i][1]/(1-b2**t))
        m.W[i]-=lr*mh[0]/(np.sqrt(vh[0])+eps);m.b[i]-=lr*mh[1]/(np.sqrt(vh[1])+eps)

def nadam(m,gW,gb,s,lr=0.001,b1=0.9,b2=0.999,eps=1e-8):
    s.setdefault('t',0);s['t']+=1;t=s['t']
    s.setdefault('m',[(np.zeros_like(w),np.zeros_like(b)) for w,b in zip(m.W,m.b)])
    s.setdefault('v',[(np.zeros_like(w),np.zeros_like(b)) for w,b in zip(m.W,m.b)])
    for i in range(3):
        s['m'][i]=(b1*s['m'][i][0]+(1-b1)*gW[i],b1*s['m'][i][1]+(1-b1)*gb[i])
        s['v'][i]=(b2*s['v'][i][0]+(1-b2)*gW[i]**2,b2*s['v'][i][1]+(1-b2)*gb[i]**2)
        vh=(s['v'][i][0]/(1-b2**t),s['v'][i][1]/(1-b2**t))
        mn=(b1*s['m'][i][0]/(1-b1**(t+1))+(1-b1)*gW[i]/(1-b1**t),b1*s['m'][i][1]/(1-b1**(t+1))+(1-b1)*gb[i]/(1-b1**t))
        m.W[i]-=lr*mn[0]/(np.sqrt(vh[0])+eps);m.b[i]-=lr*mn[1]/(np.sqrt(vh[1])+eps)

np.random.seed(42)
N=5000;X=np.vstack([np.random.randn(500,784)*0.3+np.random.randn(784)*0.5 for _ in range(10)])
y=np.repeat(np.arange(10),500)
y=OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1,1))
X=(X-X.min())/(X.max()-X.min())
Xt,Xv,yt,yv=train_test_split(X,y,test_size=0.2,random_state=42)

print('SGD...');h1=run(sgd)
print('Adam...');h2=run(adam)
print('Nadam...');h3=run(nadam)

fig,axes=plt.subplots(1,2,figsize=(12,5))
for h,n,c in zip([h1,h2,h3],['SGD','Adam','Nadam'],['#ff7b72','#79c0ff','#56d364']):
    axes[0].plot(h['vl'],color=c,label=n,linewidth=2)
    axes[1].plot(h['va'],color=c,label=n,linewidth=2)
axes[0].set_title('Loss Validation');axes[1].set_title('Accuracy Validation')
for ax in axes: ax.legend();ax.grid(alpha=0.3)
plt.suptitle('SGD vs Adam vs Nadam',fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig('comparaison.png',dpi=120);plt.show()
