### **Proyecto Final de Machine Learning**

El objetivo de este proyecto es desarrollar un modelo de machine learning, desde la obtención de datos hasta su despliegue.

La entrega será un **repositorio de github** con el desarrollo del proyecto: adquisición de datos, limpieza, EDA, feature engineering, modelado de datos, iteración de modelos, evaluación de modelos, interpretación de modelos, impacto en negocio.

El repositorio deberá mostrar una estructura similar a la siguiente:

```
|-- nombre_proyecto_final_ML
    |-- notebooks
    |   |-- 01_Fuentes.ipynb
    |   |-- 02_LimpiezaEDA.ipynb
    |   |-- 03_Entrenamiento_Evaluacion.ipynb
    |   |-- ...
    |
    |-- src
    |   |-- preprocessing.py
    |   |-- training.py
    |   |-- evaluation.py
    |   |-- ...
    |
    |-- models
    |   |-- trained_model.pkl
    |   |-- model_config.yaml
    |   |-- ...
    |
    |-- app
    |   |-- app.py
    |   |-- requirements.txt
    |   |-- ...
    |
    |-- docs
    |   |-- negocio.ppt
    |   |-- ds.ppt
    |   |-- memoria.md
    |   |-- ...
    |
    |
    |-- README.md
    |-- ...

```

**Instrucciones**
Hemos usado Anaconda para crear el entorno y añadir las dependencias.
El archivo requirements.txt contiene las dependencias para la "app prototype".