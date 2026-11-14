from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
Path('resultados').mkdir(exist_ok=True)
plt.rcParams.update({'figure.figsize':(9,5),'axes.spines.top':False,'axes.spines.right':False})
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def ejecutar_proyecto():
    rng=np.random.default_rng(42)
    X=np.vstack([rng.normal([1800,350],[200,80],(60,2)),rng.normal([4500,1400],[300,180],(60,2)),rng.normal([8000,900],[400,130],(60,2))])
    np.savetxt('datos.csv',X,delimiter=',',header='ingreso,gasto',comments='')
    scaler=StandardScaler(); Z=scaler.fit_transform(X); scores={}; models={}
    for k in range(2,7):
        model=KMeans(n_clusters=k,n_init=20,random_state=42,max_iter=300).fit(Z)
        models[k]=model; scores[k]={'silueta':float(silhouette_score(Z,model.labels_)),'inercia':float(model.inertia_)}
    k=max(scores,key=lambda v:scores[v]['silueta']); model=models[k]; centers=scaler.inverse_transform(model.cluster_centers_)
    print('Comparación:',json.dumps(scores,indent=2)); print('K elegido:',k,'Iteraciones:',model.n_iter_)
    for g,center in enumerate(centers): print('Grupo',g,'cantidad',int((model.labels_==g).sum()),'centro ingreso/gasto',center)
    print('Grupo de nuevo cliente (4600,1500):',model.predict(scaler.transform([[4600,1500]]))[0])
    fig,ax=plt.subplots(1,2,figsize=(12,4)); ax[0].plot(list(scores),[v['silueta'] for v in scores.values()],'o-'); ax[0].set(xlabel='K',ylabel='Silueta',title='Selección de grupos')
    ax[1].scatter(X[:,0],X[:,1],c=model.labels_,cmap='viridis'); ax[1].scatter(centers[:,0],centers[:,1],marker='X',c='coral',s=180,edgecolor='black'); ax[1].set(xlabel='Ingreso',ylabel='Gasto',title='Clientes y centroides')
    plt.tight_layout(); plt.savefig('resultados/proyecto.png',dpi=140); plt.show()
    np.savetxt('resultados/segmentos.csv',np.c_[X,model.labels_],delimiter=',',header='ingreso,gasto,grupo',comments='')
    display(Markdown(f"### Interpretación de la ejecución\nLa mejor silueta entre los candidatos corresponde a **{k} grupos**, con **{scores[k]['silueta']:.3f}**. Los centroides describen perfiles relativos; sus identificadores no tienen orden."))
    return {'k_elegido':k,'comparacion':scores,'centros':centers.tolist()}

if __name__=='__main__':
    metricas=ejecutar_proyecto()
    Path('resultados/metricas.json').write_text(json.dumps(metricas,indent=2),encoding='utf-8')
