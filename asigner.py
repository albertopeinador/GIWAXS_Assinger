#import fityk_export as fe
import streamlit as st
import only_sess as ses
import plotly.graph_objects as go


st.set_page_config(layout="wide")
ctrls, plot = st.columns([1, 2])

check = False

# Apply custom CSS to hide the "Drag and drop file here" text
st.markdown("""
    <style>
        span.st-emotion-cache-9ycgxx {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

#st.markdown("""
#    <style>
#        small.st-emotion-cache-1aehpvj {
#            display: none !important;
#        }
#    </style>
#""", unsafe_allow_html=True)

st.markdown("""
    <style>
        /* Center and bold the file uploader label */
        div[data-testid="stFileUploader"] label {
            font-weight: 900 !important;  /* Strongest bold */
            text-align: center !important;
            display: block !important;
            font-size: 1.1em !important; /* Slightly larger for emphasis */
        }
    </style>
""", unsafe_allow_html=True)

with ctrls:
    #fit, peaks = st.columns(2)
    fig1 = go.Figure()
    with st.expander('Input controls', expanded=True):
        session = st.file_uploader('Session File', type=['.fit'])
        #with peaks:
        #    peaks = st.file_uploader('Peak parameters', type=['.peaks'])
    
        if session is not None:
            expected_peaks = st.text_input('Expected peaks in the samples', placeholder='Peaks with x will be assigned but ignored. Use ; as separator')
            use_peaks = st.text_input('Names of peak functions used', placeholder='The names of the fityk functions used to fit peaks. Use ; as separator')
            if expected_peaks != '':
                ex_peak_list = expected_peaks.split(';')
            if use_peaks != '':
                use_peak_list = use_peaks.split(';')
            #st.write(ex_peak_list)
            #one_head = st.checkbox('Only use one header in output file')
            ignore_peaks = st.checkbox('Assign  unknown peaks', value=False)
            #assign_strength = st.number_input('proximity threshold fos assignation', value = 0.05, step=.01)
            if expected_peaks and use_peaks != '':
                names, asigns, data, models, funcs, VARS = ses.asigner(session, ex_peak_list, ignore_peaks, use_peak_list)
                filter_plot = st.multiselect('First Assignation plot filters', ['data', 'model', 'labels'], default = ['data', 'model', 'labels'])
                firstkey = list(key for key, val in names.items() if val == list(asigns.keys())[0])[0]
                #high_vals = [(idx, val) for idx, val in enumerate(VARS) if val > 100]
                #if high_vals != []:
                #    warnings.warn(f'Warning!!!! high values found in variables {high_vals}')
                X = [data[firstkey][j][0] for j in range(len(data[firstkey]))]
                Y = [data[firstkey][j][1] for j in range(len(data[firstkey]))]
                if 'data' in filter_plot:
                    fig1.add_trace(go.Scatter(x = X, y = Y))
                
                if 'model' in filter_plot:
                    #st.write(models[firstkey])
                    x, y = ses.calcmodel(models[firstkey], funcs, VARS)
                    fig1.add_trace(go.Scatter(x = x, y = y))
                    #st.write(asigns[names[firstkey]])
                if 'labels' in filter_plot:
                    for plane in asigns[names[firstkey]]:
                        fig1.add_annotation(
                            x=asigns[names[firstkey]][plane][1],  
                            y=ses.find_closest_value(X, Y, asigns[names[firstkey]][plane][1]),  # Arrowhead position

                            ax=asigns[names[firstkey]][plane][1] + 3,  # Move annotation text right
                            ay=ses.find_closest_value(X, Y, asigns[names[firstkey]][plane][1]) + .1 * max(Y),  # Move annotation text up

                            xref="x", yref="y",
                            axref="x", ayref="y",

                            text=plane,  # Annotation text
                            showarrow=True,
                            arrowhead=6,
                            arrowsize=1.5,
                            arrowwidth=1,
                            arrowcolor="gray"
                        )

    if session is not None:
        if expected_peaks and use_peaks != '':
            check = True
            fig1.update_layout( showlegend = False, title_text = 'First Assignment', title_x = .3)
            st.plotly_chart(fig1, use_container_width=True)

            st.download_button('Download csv', data = ses.write_csv(asigns), file_name='fitting_parameters.csv')

if check:
    fig2 = go.Figure()
    with plot:
        sample, filters = st.columns([1, 1.5])
        with sample:
            selected = st.selectbox('Plot sample: ', list(asigns.keys()))
        with filters:
            filts = st.multiselect('Plot filters', ['sample data', 'full model', 'peak labels'], default=['sample data', 'full model', 'peak labels'])
        plotkey = list(key for key, val in names.items() if val == selected)[0]
        X = [data[plotkey][j][0] for j in range(len(data[plotkey]))]
        Y = [data[plotkey][j][1] for j in range(len(data[plotkey]))]
        if 'sample data' in filts:
            fig2.add_trace(go.Scatter(x = X, y = Y))
        
        if 'full model' in filts:
            #st.write(models[firstkey])
            x, y = ses.calcmodel(models[plotkey], funcs, VARS)
            fig2.add_trace(go.Scatter(x = x, y = y))
            #st.write(asigns[names[firstkey]])
        if 'peak labels' in filts:
            for plane in asigns[names[plotkey]]:
                fig2.add_annotation(
                    x=asigns[names[plotkey]][plane][1],  
                    y=ses.find_closest_value(X, Y, asigns[names[plotkey]][plane][1]),  # Arrowhead position
                    ax=asigns[names[plotkey]][plane][1] + 3,  # Move annotation text right
                    ay=ses.find_closest_value(X, Y, asigns[names[plotkey]][plane][1]) + .1 * max(Y),  # Move annotation text up
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    text=plane,  # Annotation text
                    showarrow=True,
                    arrowhead=6,
                    arrowsize=1.5,
                    arrowwidth=1,
                    arrowcolor="gray"
                )
        fig2.update_layout(height=600, showlegend = False, title_text=f'{selected} Assignment', title_x=0.3)
        st.plotly_chart(fig2, use_container_width=True)
        #st.write(ses.write_csv(asigns))



#if session is not None and peaks is not None:
#    with plot:
#        sample, datacheck, modelcheck = st.columns(3)
#        with sample:
#            selected = st.selectbox('Plot sample: ', ['test', 'other'])
#
    


