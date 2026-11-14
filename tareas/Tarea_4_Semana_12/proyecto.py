from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
Path('resultados').mkdir(exist_ok=True)
plt.rcParams.update({'figure.figsize':(9,5),'axes.spines.top':False,'axes.spines.right':False})
from scipy import ndimage
from matplotlib.patches import Rectangle

def ejecutar_proyecto():
    yy,xx=np.mgrid[:160,:240]; image=np.zeros((160,240))
    image[(xx-45)**2+(yy-45)**2<=22**2]=1; image[25:65,105:145]=1
    image[(xx-190)**2+(yy-115)**2<=25**2]=1; image[100:140,30:70]=1
    plt.imsave('imagen_entrada.png',image,cmap='gray',vmin=0,vmax=1)
    labels,total=ndimage.label(image>.5); detections=[]
    fig,ax=plt.subplots(1,3,figsize=(15,4)); ax[0].imshow(image,cmap='gray'); ax[0].set_title('Entrada sintética'); ax[1].imshow(labels,cmap='nipy_spectral'); ax[1].set_title('Componentes'); ax[2].imshow(image,cmap='gray'); ax[2].set_title('Reconocimiento')
    for i,(sy,sx) in enumerate(ndimage.find_objects(labels),1):
        area=int((labels[sy,sx]==i).sum()); h=sy.stop-sy.start; w=sx.stop-sx.start; ratio=area/(h*w)
        kind='cuadrado' if ratio>.88 else 'circulo'
        detections.append({'id':i,'clase':kind,'x':sx.start,'y':sy.start,'ancho':w,'alto':h,'area':area,'ocupacion':ratio})
        ax[2].add_patch(Rectangle((sx.start-.5,sy.start-.5),w,h,fill=False,edgecolor='lime',linewidth=2)); ax[2].text(sx.start,sy.start-4,kind,color='lime',fontsize=9,bbox={'facecolor':'black','pad':1})
    for a in ax:a.axis('off')
    plt.tight_layout(); plt.savefig('resultados/proyecto.png',dpi=140); plt.show()
    counts={k:sum(d['clase']==k for d in detections) for k in ['circulo','cuadrado']}
    assert total==4 and counts=={'circulo':2,'cuadrado':2}
    print(json.dumps(detections,indent=2))
    display(Markdown(f"### Interpretación de la ejecución\nSe detectaron **{total} figuras**, dos de cada clase, en coincidencia con la escena construida. Esta comprobación verifica el caso controlado; no mide precisión en fotografías independientes."))
    return {'total':int(total),'conteos':counts,'detecciones':detections}

if __name__=='__main__':
    metricas=ejecutar_proyecto()
    Path('resultados/metricas.json').write_text(json.dumps(metricas,indent=2),encoding='utf-8')
