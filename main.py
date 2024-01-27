import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from streamlit.web import cli as stcli

st.write('PTSD Class Dashboard')

df = pd.read_csv('C:\BACKUP DATA HENDRI ADI\TA\PTSD.csv')
st.subheader('user input features')
st.write(df)
st.sidebar.header('Fill the form')

def form_param():
    p1=st.sidebar.slider('1. Memiliki kenangan yang berulang, mengganggu, dan tidak diinginkan dari pengalaman stress akibat kekerasan orang tua?', min_value=0,max_value=4, step=1)
    p2=st.sidebar.slider('2. Mimpi yang berulang dan mengganggu tentang pengalaman stres?', min_value=0,max_value=4, step=1)
    p3=st.sidebar.slider('3. Tiba-tiba merasa atau bertindak seolah-olah pengalaman stres itu benar-benar terjadi lagi ketika mendapat trigger (seolah-olah anda benar-benar kembali ke masa kejadian traumatis anda)?', min_value=0,max_value=4, step=1)
    p4=st.sidebar.slider('4. Merasa sangat kesal/marah/takut/cemas ketika sesuatu mengingatkan anda pada pengalaman stres/trauma?', min_value=0,max_value=4, step=1)
    p5=st.sidebar.slider('5. Memiliki reaksi fisik yang kuat ketika sesuatu mengingatkan anda pada pengalaman stres/trauma yang diakibatkan oleh perilaku orang tua\n (misalnya, jantung berdebar, kesulitan bernapas, berkeringat)?', min_value=0,max_value=4, step=1)
    p6 =st.sidebar.slider('6. Menghindari ingatan, pikiran, atau perasaan yang berhubungan dengan pengalaman stres/trauma yang diakibatkan oleh perilaku orang tua?', min_value=0,max_value=4, step=1)
    p7=st.sidebar.slider('7. Menghindari pengingat eksternal dari pengalaman stres/trauma (contoh: orang, tempat, percakapan, aktivitas, objek, atau situasi)?', min_value=0,max_value=4, step=1)
    p8=st.sidebar.slider('8. Kesulitan mengingat bagian yang penting dari pengalaman stres/trauma?', min_value=0,max_value=4, step=1)
    p9=st.sidebar.slider('9. Memiliki keyakinan negatif yang kuat tentang diri sendiri, orang lain, atau dunia \n (misalnya, memiliki pemikiran seperti: saya merasa buruk, ada sesuatu yang salah dengan saya, tidak ada yang bisa dipercaya, saya merasa gagal)?', min_value=0,max_value=4, step=1)
    p10=st.sidebar.slider('10. Menyalahkan diri sendiri atau orang lain untuk pengalaman stres/trauma atau apa yang terjadi setelah itu?', min_value=0,max_value=4, step=1)
    p11=st.sidebar.slider('11. Memiliki perasaan negatif yang kuat seperti takut, cemas, marah, bersalah, atau malu?', min_value=0,max_value=4, step=1)
    p12=st.sidebar.slider('12. Kehilangan minat pada aktivitas yang dulu anda sukai?', min_value=0,max_value=4, step=1)
    p13=st.sidebar.slider('13. Merasa jauh atau terputus dari orang lain (terutama keluarga atau teman)?', min_value=0,max_value=4, step=1)
    p14=st.sidebar.slider('14. Kesulitan mengalami perasaan positif (misalnya, menjadi tidak dapat merasakan kebahagiaan atau memiliki perasaan cinta untuk orang-orang dekat denganmu)?', min_value=0,max_value=4, step=1)
    p15=st.sidebar.slider('15. Perilaku mudah tersinggung, ledakan kemarahan, atau bertindak agresif?', min_value=0,max_value=4, step=1)
    p16=st.sidebar.slider('16. Mengambil terlalu banyak risiko atau melakukan hal-hal yang dapat menyebabkan menyakiti anda (diri sendiri)?', min_value=0,max_value=4, step=1)
    p17=st.sidebar.slider('17. Menjadi "sangat waspada" atau berhati-hati', min_value=0,max_value=4, step=1)
    p18=st.sidebar.slider('18. Merasa gelisah atau mudah terkejut', min_value=0,max_value=4, step=1)
    p19=st.sidebar.slider('19. Mengalami kesulitan dalam berkonsentrasi', min_value=0,max_value=4, step=1)
    p20=st.sidebar.slider('20. Kesulitan untuk tidur atau sering insomnia', min_value=0,max_value=4, step=1)

    data = {
        'pertanyaan ke-1':p1,
        'pertanyaan ke-2':p2,
        'pertanyaan ke-3':p3,
        'pertanyaan ke-4':p4,
        'pertanyaan ke-5':p5,
        'pertanyaan ke-6':p6,
        'pertanyaan ke-7':p7,
        'pertanyaan ke-8':p8,
        'pertanyaan ke-9':p9,
        'pertanyaan ke-10':p10,
        'pertanyaan ke-11':p11,
        'pertanyaan ke-12':p12,
        'pertanyaan ke-13':p13,
        'pertanyaan ke-14':p14,
        'pertanyaan ke-15':p15,
        'pertanyaan ke-16':p16,
        'pertanyaan ke-17':p17,
        'pertanyaan ke-18':p18,
        'pertanyaan ke-19':p19,
        'pertanyaan ke-20':p20
    }
    features = pd.DataFrame(data, index=[0])
    return features

dataset = form_param()
cluster = ({'ClusterB':df.iloc[:,0:5].sum(axis=1).values,
           'ClusterC':df.iloc[:,5:7].sum(axis=1).values,
           'ClusterD':df.iloc[:,7:14].sum(axis=1).values,
           'ClusterE':df.iloc[:,14:20].sum(axis=1).values,
            'Diagnosis':df.iloc[:, -1].values
})

df_new = pd.DataFrame(cluster,columns=['ClusterB','ClusterC','ClusterD','ClusterE','Diagnosis'])
st.write(df_new)
#clusters = st.multiselect('select cluster:',options=df_new.columns[0:4].tolist(), max_selections=2, default=['ClusterB','ClusterC'])
col_x = st.selectbox('select side x', df_new.columns[0:4])
col_y = st.selectbox('select side y', df_new.columns[0:4])

max_values = {'ClusterB': 20, 'ClusterC': 8, 'ClusterD': 28, 'ClusterE': 24}
total_sum = df_new[['ClusterB', 'ClusterC', 'ClusterD', 'ClusterE']].clip(upper=max_values)
total_sum = total_sum.sum()
percentage_values = total_sum / total_sum.sum() * 100

#diagnos = df_new[(df_new['Diagnosis'])][cluster]
#fig=px.scatter(df_new, x=col_x,y=col_y, color='Diagnosis', color_discrete_map={'0': 'blue', '1': 'red'})
#st.plotly_chart(fig)
feature = st.selectbox('Which feature?', df_new.columns[0:4])
plot_types = st.multiselect('Select plot types:', ['Scatter Plot', 'Bar Plot', 'Histogram', 'Pie Chart'])
#fig2 = px.histogram(df_new,x=feature,color='Diagnosis')
for plot_type in plot_types:
    if plot_type == 'Scatter Plot':
        fig = px.scatter(df_new, x=col_x, y=col_y, color='Diagnosis', color_discrete_map={'0': 'blue', '1': 'red'})
    elif plot_type == 'Bar Plot':
        fig = px.bar(df_new, x=col_x, color='Diagnosis', color_discrete_map={'0': 'blue', '1': 'red'})
        fig.update_traces(text=df_new[col_y], textposition='outside')
        fig.update_layout(barmode='relative')
    elif plot_type == 'Histogram':
        fig = px.histogram(df_new, x=col_x, color='Diagnosis', color_discrete_map={'0': 'blue', '1': 'red'})
        #fig.add_trace(px.scatter(df_new, x=col_x, y=col_y, color='Diagnosis', color_discrete_map={'0': 'blue', '1': 'red'}).data[0])
        fig.update_layout(barmode='group', showlegend=True)
    elif plot_type == 'Pie Chart':
        # Create a Pie Chart with Percentage of Max Values
        pie_data = pd.DataFrame({'cluster': percentage_values.index, 'percentage': percentage_values.values})
        fig = px.pie(
            pie_data,
            names='cluster',
            values='percentage',
            title='Percentage of Max Values for Each Cluster'
        )
        #fig = px.pie(df_new, labels=cluster, names= percentage_values,values=percentage_values,title='Percentage of Max Values for Each Cluster')
    else:
        st.warning(f"Invalid plot type selected: {plot_type}")

    st.plotly_chart(fig)

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
clf = svm.SVC(kernel='linear')
clf.fit(x,y)

pred = clf.predict(dataset)
comparing = clf.predict(x)
result_df = pd.DataFrame({'True Labels':y,'Predict':comparing})
st.subheader('Comparing Predict')
st.write(result_df)

st.subheader('predict')
st.write(pred)