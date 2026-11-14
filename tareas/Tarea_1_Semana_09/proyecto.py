from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
Path('resultados').mkdir(exist_ok=True)
plt.rcParams.update({'figure.figsize':(9,5),'axes.spines.top':False,'axes.spines.right':False})
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def ejecutar_proyecto():
    rng=np.random.default_rng(42)
    X=np.column_stack([rng.uniform(0,10,240),rng.uniform(50,100,240)])
    y=2+1.1*X[:,0]+.06*X[:,1]+rng.normal(0,.8,240)
    np.savetxt('datos.csv',np.c_[X,y],delimiter=',',header='horas,asistencia,nota',comments='')
    a,b,c,d=train_test_split(X,y,test_size=.2,random_state=42)
    simple=LinearRegression().fit(a[:,[0]],c)
    multiple=LinearRegression().fit(a,c)
    pred={'Base':np.full_like(d,c.mean()),'Simple':simple.predict(b[:,[0]]),'Multiple':multiple.predict(b)}
    m={k:{'MAE':float(mean_absolute_error(d,v)),'RMSE':float(np.sqrt(mean_squared_error(d,v))),'R2':float(r2_score(d,v))} for k,v in pred.items()}
    print(json.dumps(m,indent=2)); print('Intercepto y coeficientes múltiples:',multiple.intercept_,multiple.coef_)
    print('Predicción: 6 horas y 90 % asistencia:',multiple.predict([[6,90]])[0])
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    ax[0].scatter(b[:,0],d); order=np.argsort(b[:,0]); ax[0].plot(b[order,0],pred['Simple'][order],color='coral')
    ax[0].set(xlabel='Horas',ylabel='Nota',title='Regresión simple en prueba')
    ax[1].scatter(d,pred['Multiple']); ax[1].plot([d.min(),d.max()],[d.min(),d.max()],'--',color='coral')
    ax[1].set(xlabel='Nota real',ylabel='Predicción',title='Regresión múltiple')
    plt.tight_layout(); plt.savefig('resultados/proyecto.png',dpi=140); plt.show()
    plt.scatter(pred['Multiple'],d-pred['Multiple']); plt.axhline(0,color='coral'); plt.xlabel('Predicción'); plt.ylabel('Residuo'); plt.title('Diagnóstico de residuos')
    plt.savefig('resultados/residuos.png',dpi=140); plt.show()
    np.savetxt('resultados/predicciones.csv',np.c_[d,pred['Simple'],pred['Multiple']],delimiter=',',header='real,simple,multiple',comments='')
    display(Markdown(f"### Interpretación de la ejecución\nMAE múltiple: **{m['Multiple']['MAE']:.3f}** puntos; simple: **{m['Simple']['MAE']:.3f}**; base: **{m['Base']['MAE']:.3f}**. La asistencia aporta información porque interviene en la generación de los datos."))
    return m

if __name__=='__main__':
    metricas=ejecutar_proyecto()
    Path('resultados/metricas.json').write_text(json.dumps(metricas,indent=2),encoding='utf-8')
