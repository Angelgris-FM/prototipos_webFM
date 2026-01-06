import streamlit as st
from scipy.io import loadmat
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # para la grafica tridimensional interactiva
# usaremos la siguiente libreria
import matplotlib.pyplot as plt
from PIL import Image
datos = loadmat("TempMed.mat")

#print(datos.keys()) para ver que variables contiene ,seleccionamos la que contiene datos 
# es decir el nombre de la variable vista en matlab
Temp=datos['T2']
# convertimos el tipo de datos en df y array
df_temp=pd.DataFrame(Temp)
np_temp=np.array(Temp)


# parametros

coef_dif = 22.8e-6
hx = 0.05
ht = 60
x = np.arange(0, 1+hx, hx)
t = np.arange(0, 3600+ht, ht)

# ---- funciones graficas----------



# objetivo es crear la grafica tridimensional tal que se pueda rotar y hacer zoom 
# como en matplotlib o un plot 3d en matlab
def grafico_tridimensional(X, T, Z):
    X, T = np.meshgrid(X, T)               
    fig2 = go.Figure(data=[go.Surface(
        x=X ,y=T ,z=Z, colorscale='jet',
        colorbar=dict(
            tickfont=dict(color='black',size=12),
            ticks="outside"
        ))])
    # customizar y editar por ejemplo el fondo 
    # los colores de las etiquetas etc
    
    fig2.update_layout(
        # este apartado coloca un titulo general a la figura
        title=dict(text="Gráfica tridimensional de temperaturas",font=dict(color='black'),
                x=0.5,
                y=0.95),
        # configuramos como y que detalles se presentan en la grafica tridimensional
        scene=dict(
            xaxis=dict(title=dict(text='posicion en x [m]',font=dict(color='black'))
                    ,tickfont=dict(color='black')),
            yaxis=dict(title=dict(text='tiempo [s]',font=dict(color='black')),
                    tickfont=dict(color='black')),
            zaxis=dict(title=dict(text="Temperatura [C]",font=dict(color="black")),
                    tickfont=dict(color="black")),
            aspectmode='cube'        # mantiene proprociones iguales en x ,y ,z
            ),
        # configuramos el color de fondo y el tamaño de la figura
        paper_bgcolor='white',
        width=900,    # ancho
        height=700    # alto 
    )                    # hace que el grafico se adapte al ancho de contenedor  
    st.plotly_chart(fig2,use_container_width=True)


def grafico_promedio(df, tiempo):
    fig, ax = plt.subplots()
    ax.plot(tiempo, df['T promedio [C]'], color='blue', marker='o', linestyle='None')
    ax.set_xlabel('tiempo [s]')
    ax.set_ylabel('Temperatura promedio [C]')
    ax.set_title('Temperatura en cada instante de tiempo')
    st.pyplot(fig)
# para la animacion 2_d en la web usando streamlit resulta mas practico ,
# sencillo y "entendible" usando la libreria pandas (dataframes) + plotly


def animacion_T(df_temp,x,t):
    # df:temperaturas
    # x:posiciones
    # t:tiempos
    
    # convertir a formato largo
    
    df_long= pd.DataFrame(df_temp.values,columns=x)   # columnas = posiciones
    df_long["t"]= t                                   # agregar la columna tiempo
    # reoganizar con melt
    df_long =df_long.melt(id_vars="t", var_name="x", value_name="Temp")
    # id_vars : mantiene la columna de tiempo fija
    # var_name : convierte los nombres de las columnas(que son las posiciones en x ) en una columna llamada X
    # value:name : los valores de temperatura quedan en ua columna llamada "Temp"
    
    # una vez hecho esto procedemos a crear la animacion con plotly
    fig =px.scatter(df_long,x="x",y="Temp",
                    animation_frame="t",              #cada tiempo es un frame
                    color='Temp',                      # color dinamico segun la temperatura
                    color_continuous_scale="jet",   # paleta de colores (rojo-amarillo-azul)
                    range_y=[df_long["Temp"].min(),df_long["Temp"].max()],
                    labels={"x":"Posición en x [m]", "Temp":"Temperatura [C]", "t":"Tiempo [s]"},
                    title="Evolución de la temperatura (nodos)")
    # Fondo blanco
    fig.update_layout(
        plot_bgcolor="white",          # fondo del area del trazadp
        paper_bgcolor="#00CED1",         # fondo del lienzo completo
        # 
        title=dict(font=dict(color="black")),
        xaxis=dict(
        title=dict(font=dict(color="black")),
        tickfont=dict(color="black")
        ),
        yaxis=dict(
        title=dict(font=dict(color="black")),
        tickfont=dict(color="black")
        ),
        coloraxis_colorbar=dict(
        tickfont=dict(color="black"),
        title=dict(font=dict(color="black"))
        )
    )

    
    st.plotly_chart(fig, use_container_width=True)
    
def temperatura_promedio(time): 
    df_temp['T promedio [C]'] = df_temp.mean(axis=1)  # saca la media de cada fila de mi set de datos originales es decir la temperatura media en cada instante de tiempo
    df_temp['tiempo [s]'] = df_temp.index * ht   # como estamos manejando el formato dataframe añadimos una nueva columna con los instantes exactos de tiempo 
    st.write(df_temp[['tiempo [s]', 'T promedio [C]']])
    if st.button("Mostrar gráfico promedio"):
        grafico_promedio(df_temp, time)
# --- Interfaz principal ---
st.title("Simulación de evolución de  temperaturas  durante la fabricación del  acero")

st.write("""El siguiente proyecto de caracter ingenieril tiene como objetivo
            analizar y visualizar la historia termica de una barra de metal( de 1 metro)
            durante la fabricación del acero en cada etapa usando un modelo simplificado que sigue
            la Ecuacion diferencial parcial con condiciones de neumann en los bordes
            (sin flujo de calor en los extremos de la barra)
        """)
# añadiremos una imagen para ponernos en contexto
imagen_edp=Image.open("ecuacion_edp2.png")
imagen_edp = imagen_edp.resize((400, 400))   # ancho=600, alto=400 píxeles
st.image(imagen_edp)

opcion = st.radio(
    "Selecciona una opción:",
    (   "Visualizar evolución de la temperatura",
        "Visualizar temperatura promedio",
        "Visualizar gráfica de superficie")
)

if opcion == "Visualizar evolución de la temperatura":
    # para una verdadera animacion mas fluida en la web usaremos el formato
    # Dataframe
    animacion_T(df_temp,x,t)

elif opcion == "Visualizar temperatura promedio":
    temperatura_promedio(t)

elif opcion == "Visualizar gráfica de superficie":
    grafico_tridimensional(x, t, np_temp)
    
    
    
if st.button("Mostrar mediciones de temperaturas(set de datos base)"):

    st.write(df_temp)

