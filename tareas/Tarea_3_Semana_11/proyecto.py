from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
Path('resultados').mkdir(exist_ok=True)
plt.rcParams.update({'figure.figsize':(9,5),'axes.spines.top':False,'axes.spines.right':False})
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

def ejecutar_proyecto():
    tf.keras.utils.set_random_seed(42)
    X,y=make_circles(n_samples=600,factor=.45,noise=.08,random_state=42)
    np.savetxt('datos.csv',np.c_[X,y],delimiter=',',header='x1,x2,clase',comments='')
    a,t,b,u=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    a,v,b,w=train_test_split(a,b,test_size=.25,stratify=b,random_state=42)
    scaler=StandardScaler().fit(a)
    model=tf.keras.Sequential([tf.keras.Input(shape=(2,)),tf.keras.layers.Dense(16,activation='relu'),tf.keras.layers.Dense(8,activation='relu'),tf.keras.layers.Dense(1,activation='sigmoid')])
    model.compile(optimizer=tf.keras.optimizers.Adam(.01),loss='binary_crossentropy',metrics=['accuracy']); model.summary()
    options=tf.data.Options(); options.threading.private_threadpool_size=1
    train=tf.data.Dataset.from_tensor_slices((scaler.transform(a).astype('float32'),b.astype('float32'))).shuffle(len(b),seed=42).batch(32).with_options(options)
    valid=tf.data.Dataset.from_tensor_slices((scaler.transform(v).astype('float32'),w.astype('float32'))).batch(32).with_options(options)
    history=model.fit(train,validation_data=valid,epochs=100,verbose=0,callbacks=[tf.keras.callbacks.EarlyStopping(patience=12,restore_best_weights=True)])
    prob=model(scaler.transform(t).astype('float32'),training=False).numpy().ravel(); pred=(prob>=.5).astype(int)
    acc=float(accuracy_score(u,pred)); cm=confusion_matrix(u,pred)
    print('Exactitud de prueba:',acc); print('Matriz (filas reales, columnas predichas):',cm); print(classification_report(u,pred,zero_division=0))
    fig,ax=plt.subplots(1,2,figsize=(12,4)); ax[0].plot(history.history['loss'],label='Entrenamiento'); ax[0].plot(history.history['val_loss'],label='Validación'); ax[0].legend(); ax[0].set(xlabel='Época',ylabel='Pérdida',title='Aprendizaje')
    xx,yy=np.meshgrid(np.linspace(-1.5,1.5,120),np.linspace(-1.5,1.5,120)); zz=model(scaler.transform(np.c_[xx.ravel(),yy.ravel()]).astype('float32'),training=False).numpy().reshape(xx.shape)
    ax[1].contourf(xx,yy,zz,levels=np.linspace(0,1,11),cmap='coolwarm',alpha=.5); ax[1].scatter(t[:,0],t[:,1],c=u,cmap='coolwarm',edgecolors='black'); ax[1].set(xlabel='x1',ylabel='x2',title='Frontera y puntos de prueba')
    plt.tight_layout(); plt.savefig('resultados/proyecto.png',dpi=140); plt.show()
    model.save('resultados/modelo.keras'); np.savez('resultados/escalador.npz',media=scaler.mean_,escala=scaler.scale_)
    # Exportación para inferencia real en JavaScript, sin volver a entrenar en la web.
    web={'pesos':[z.tolist() for z in model.get_weights()],'media':scaler.mean_.tolist(),'escala':scaler.scale_.tolist()}
    Path('resultados/red_web.json').write_text(json.dumps(web),encoding='utf-8')
    display(Markdown(f"### Interpretación de la ejecución\nExactitud: **{acc:.1%}** en **{len(u)}** puntos de prueba tras **{len(history.history['loss'])} épocas**. La frontera curva refleja una relación no lineal; exactitud no equivale a calibración de probabilidades."))
    return {'exactitud':acc,'matriz_confusion':cm.tolist(),'epocas':len(history.history['loss']),'parametros':model.count_params()}

if __name__=='__main__':
    metricas=ejecutar_proyecto()
    Path('resultados/metricas.json').write_text(json.dumps(metricas,indent=2),encoding='utf-8')
