# RedesNeuronales

La red neuronal implementada logra de disminuir la perdida de entrenamiento y validación en cada epoca de entrenamiento en el conjunto de datos utilizado, 
sin embargo, al mirar la matriz de confusión obtenida, se puede notar que los resultados no son correctos, la red asocia siempre la misma clase
a cada ejemplo, esto puede deberse a varios motivos.

- Error en el methodo feed: Un error en el método feed podria ocasionar que todos los resultados psoteriores esten erroneos,
- Error en metodo train: De existir un error en el algoritmo de backpropagation, los pesos no son actualizados de manera correcta por loq ue el resultado puede ser equivocado, la red logra disminuir la funciónloss en cada iteración por lo que se sospecha que no es el caso.
- Error en función de activación o perdida: La red entrega como resultado un nuemro real entre 0 y 1 en cada una de sus coordenadas, esto no tiene sentido para el problema propuesto debidoa que se desea estimar variables categoricas, el output del modelo debiera ser un vector codificado en onehot, es decir, tener solamente ceros, y un uno en la clase que corresponda. Usar una distinta función de activacion al final de la red, de modo de garantizar un output de este estilo podria mejorar el desempeño de la red. Esto no fue posible de testear pero se cree que es la causa del problema

![alt text](https://github.com/VicentePenaLet/RedesNeuronales/blob/master/Plot%201.png?raw=true "train curves")

En cuanto a las curvas de loss que se observan en el gráfico plot1.jpg, se puede notar que el valor de la función loss parte en un valor alto para ambos casos, y decrece estrepitosamente a medida aumentan las epocas de entrenamiento. Las mejoras parecen ser súbitas, en otras palabras, la funcón alcaza valores en losq ue se queda relativamente estable por algunas epocas, y baja mucho en algunas epocas. 

Se puede observar que la eprdida ene l conjunto de test es siempre mayor que la perdida en el conjunto de entrenamiento, esto es esperable pues los datos del conjunto de test son nuevos para la red y debe generalizar a partir de lo aprendido en el conjunto de entrenamiento.

Se puede observar ademas que la perdida en ambos casos bajo desde 0.5 hasta alrededos de 0.3 en ambos casos, mostrando una clara mejoria. 

Con esto en mente, se espera que el error no esta en el método de entrenamiento, pues este efectivamente reduce la función objetivo en cada iteración.

Respecto a las dificultades de la implementación, esta se encontro principalmente en la implemetación del algoritmo de backpropagation, puesto que este depende de calculos con matrices y calculos de derivadas con los que se tiene que tener especial cuidado con las dimensiones de cada una de las matrices que se desea operar.

Debido a que el Sistema fue implementado utilizando matrices y funciones de numpy bien optimizadas, el entrenamiento no es demasiado lento y podria ser mejorado utilizando GPU.
