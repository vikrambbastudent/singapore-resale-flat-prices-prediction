import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Singapore Flat Resale Prices",page_icon="🏨",layout="wide")

def setting_bg():
    st.markdown(f""" <style>.stApp {{
                        background:url("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBw8NDw0NDQ0NDQ0NEA0NDQ0NDQ8NDQ0NFREWFhURFRUYHSggGBolGxUVITEhJSkrLi4uFx8zODM4NygtLisBCgoKDQ0NDw0NDisZFRkrKy0tLTc3LS0rLTctLS0rKy0rLSstKysrNy0rKysrKysrKysrKysrKysrKysrKysrK//AABEIALEBHAMBIgACEQEDEQH/xAAZAAEBAQEBAQAAAAAAAAAAAAABAgADBwT/xAAeEAEBAQEAAgIDAAAAAAAAAAAAARECEjEhQQNRgf/EABcBAQEBAQAAAAAAAAAAAAAAAAEAAgT/xAAWEQEBAQAAAAAAAAAAAAAAAAAAARH/2gAMAwEAAhEDEQA/APSpPhU5Mh5dTjVxFNzFcwNDFYYoHBzFxuTKCVQKgLadONIC0MaGREYcY4E2NhapJsbFC1JNgLYUGLJJwWKCCaDY2EJqbVCxJNFihIQiosdOomkOViOnXpFLLT5ipGn06YjIOYuQyKkZKcVGwzlEyaeYeYqcg4IppFeIaELYcCbGkJxFMhLJCEsEBhxsKTY2HDiQwYsJIsTY6WJxBzsbqLsawpysEjpYmwjEWCKwIJsRXSppDn1HHv2+jqOd5hBkdZPSJHXlKNmKjSHGWm5ipBI6AwYqRoqQEYcNZESHDDYCMOHCkmRsU2JYMbxVjYjicFi8bEsT4jFjEMTjYqwVJAXgxBNTV4MKRgsXYLEHOwLowjHPpK+uU2NBzqOo6uXRDpw6YnmLgRhxoplppFyJi4CzSE8ohWNioCMOMQWYnEgymRThOMtQalsWoYMUEksqhBIsUCk0KsGIIrGsQ54OotNQSixabDBXOxzsdrHPpplXLpIjmLFJioIqBoxcTFAsqQYqAlmhBYyMUmkLFFmYhBiwQYsUAoJAWEFJxlVKAawhJAqqCE1FXUlkI6WnCKhy7da5d+yHWFPKlVFRUTIuBoriTAYqFMXAWLMCYRFIsQQizEJmZkWZmSApFQYEUoAggAhJNBoITRT1U0gVFqqjoss5dOtRSDyuVz5qlU6c1SYQ0uGJhBXFRCpQSRpBJSqVIlJBJDJFgwRYMkQwKZmGlC0NrWoMGBCazWipCpNo0spqaqopA6c+qvpz6IXFSufNXKUuVSOVslXKojVcgqh5qTEVxSNIKmDAqKTKkpk6dWE6dTrasR1hrasRGi1tSa0MyDBtGla1FotCDNoBDVFPSbSGqKqppgqajqr6rlSyeVxEquUl+S5XKRcBXKpEOgummVGkFcp1GlFcMRFSgq0o1tS1bJla1HVNqZTqWlk6yGqA1kjRaKLUlakaNQLCi0prU2i0aQeqi02otIrWt0Km9FlPVRVd1ztIXKqOXF+HSVKLlU5x00ExUqJTqK9XK5TpUrOHV6dRp1FcqnOVXkEsaNbUTrJ0pK1okpFtACUA2lNa1ootSOtqQgq1NrVJTam34bU04zrVqLU2kFFa1FpDd1z6NqOoRq+XTlyiuahHVcrnKqBpR0Qgq0xMVoJMqVIxUMTDoKtYBIwwSkJQGtqSrRazJNoask1AFKOhrEoNorWikBFrW4m0g1NFotIbUWq1y66IrC0ajosu0+lclgYvlbMGjPpU9swKhCwLRUDKmKZmCMaMyRZmSjQxmRalmCTSzFAftmSYBkBQzEJvpyvv+MxgFEZiE/kceizUZqOhWYh//9k=");
                        background-size: cover}}
                     </style>""", unsafe_allow_html=True)
setting_bg()
df1=pd.read_csv(r"C:\Users\HP\Downloads\combined.csv")
df=pd.read_csv(r'C:\Users\HP\Downloads\Proceesed_data.csv')
df=df.drop(['Unnamed: 0'],axis=1) 
tab1,tab2=st.tabs(['HOME','PREDICTION'])
with tab1:
    st.markdown("<h1 style='text-align: center;font-size: 36px; color: darkblue;'>Analysis of Singapore Flat Resale Prices</h1>", unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Singapore_Skyline_at_Bluehour.jpg/1920px-Singapore_Skyline_at_Bluehour.jpg",use_column_width=True)

with tab2:  
    st.markdown("<h1 style='text-align: center;font-size: 36px; color: darkblue;'>Analysis of Singapore Flat Resale Prices</h1>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")  
    col3,col4,col5=st.columns(3,gap="small")  

    with col3:
        town_mapping = {'ANG MO KIO': 1, 'BEDOK': 2, 'BISHAN': 3, 'BUKIT BATOK': 4, 'BUKIT MERAH': 5, 'BUKIT TIMAH': 6,
                    'CENTRAL AREA': 7, 'CHOA CHU KANG': 8, 'CLEMENTI': 9, 'GEYLANG': 10, 'HOUGANG': 11,
                    'JURONG EAST': 12, 'JURONG WEST': 13, 'KALLANG/WHAMPOA': 14, 'MARINE PARADE': 15, 'QUEENSTOWN': 16,
                    'SENGKANG': 17, 'SERANGOON': 18, 'TAMPINES': 19, 'TOA PAYOH': 20, 'WOODLANDS': 21, 'YISHUN': 22,
                    'LIM CHU KANG': 23, 'SEMBAWANG': 24, 'BUKIT PANJANG': 25, 'PASIR RIS': 26, 'PUNGGOL': 27}

        town_key = st.selectbox('**Select a town**', list(town_mapping.keys()))
        town = town_mapping[town_key] 

        streets = df1['street_name'].unique()
        
        def map_street_to_number(street_name):
            street_mapping = {street: idx + 1 for idx, street in enumerate(streets)}
            return street_mapping.get(street_name)


        selected_street = st.selectbox('Select Street Name:', streets)

        selected_street_number = map_street_to_number(selected_street)


        # get block details
        block = st.text_input('**Enter the block number (eg.201A)**', value=438)
        # Define a mapping for letters to decimal values
        letter_mapping = {chr(ord('A') + i): f'.{i + 1}' for i in range(26)}
        block_decimal = float(''.join(letter_mapping.get(c, c) for c in block))

        # Define a mapping of flat_type to numbers
        category_mapping = {
        '1 ROOM': 1,
        '2 ROOM': 2,
        '3 ROOM': 3,
        '4 ROOM': 4,
        '5 ROOM': 5,
        'EXECUTIVE': 6,
        'MULTI GENERATION': 7
        }

        flat_type = st.selectbox('**Select Flat Type**', list(category_mapping.keys()))
        flat_type_value = category_mapping[flat_type]

    with col4:
        # Flat Model
        flat_model_mapping = {'IMPROVED': 1, 'NEW GENERATION': 2, 'MODEL A': 3, 'STANDARD': 4, 'SIMPLIFIED': 5,
                        'MODEL A-MAISONETTE': 6, 'APARTMENT': 7, 'MAISONETTE': 8, 'TERRACE': 9, '2-ROOM': 10,
                        'IMPROVED-MAISONETTE': 11,
                        'MULTI GENERATION': 12, 'PREMIUM APARTMENT': 13, 'Improved': 14, 'New Generation': 15,
                        'Model A':
                            16, 'Standard': 17, 'Apartment': 18, 'Simplified': 19, 'Model A-Maisonette': 20,
                        'Maisonette':
                            21, 'Multi Generation': 22, 'Adjoined flat': 23, 'Premium Apartment': 24, 'Terrace': 25,
                        'Improved-Maisonette': 26, 'Premium Maisonette': 27, '2-room': 28, 'Model A2': 29, 'DBSS': 30,
                        'Type S1': 31, 'Type S2': 32, 'Premium Apartment Loft': 33, '3Gen': 34}

        flat_model = st.selectbox("**Select Flat Model**", list(flat_model_mapping.keys()))
        flat_model_value = flat_model_mapping[flat_model]  

        # input for area in sq.m
        floor_area = st.number_input("**Enter the area**", value=95.0)

        
        storey_lower = st.number_input("**Enter the lower bound of the storey range**", value=4,min_value=0,max_value=6)
        # Upper
        storey_upper = st.number_input("**Enter the upper bound of the storey range**", value=6, min_value=storey_lower)

    with col5:
        lease_commence_year = st.number_input("**Enter the lease commence year**", value=1990)
        remaining_lease = st.text_input("**Enter the remaining lease duration (years-months e.g., '63-7')**")

        # Initialize years and months to 0
        years = 0
        months = 0

        # Parse the input text
        if remaining_lease:
            # Split the input text using the '-' delimiter
            parts = remaining_lease.split('-')
            if len(parts) == 2:
                years = float(parts[0])
                months = float(parts[1])

        # Convert years and months to a decimal representation
        total_years = years + (months / 12)

        resale_year = st.number_input('**Enter the resale year**', value=2016, min_value=lease_commence_year,max_value=2023)
        resale_month = st.number_input("**Enter the resale month**", value=12,max_value=12)


        features = {'town':town,
            'flat_type': flat_type_value,
            'block': block_decimal,
            'street_name': selected_street_number,
            'floor_area_sqm': floor_area,
            'flat_model': flat_model_value,
            'lease_commence_date': lease_commence_year,
            'remaining_lease': total_years,
            'resale_year': resale_year,
            'resale_month': resale_month,
            'storey_lower_bound': storey_lower,
            'storey_upper_bound': storey_upper,
            }
        
        features_df = pd.DataFrame(features, index=[0])
    
    # predict the resale price
    if st.button('Predict Price'):
        
        def load_data():
            # Load data
            data = pd.read_csv(r'C:\Users\HP\Downloads\Proceesed_data.csv')
            sample_df = data.sample(n=5000, random_state=42)
            return sample_df
        c_df = load_data()

        def train_model(data):
            # Train model here
            X = data.drop(['resale_price','Unnamed: 0'], axis=1)
            y = data['resale_price']
            trained_model = RandomForestRegressor(random_state=42)
            trained_model.fit(X, y)
            return trained_model


        rf_regressor = train_model(c_df)
        prediction = rf_regressor.predict(features_df)  # Replace X_test with your test data

        predicted_price = str(prediction)[1:-1]
        #predicted_price=prediction.round(2)

        st.write(f"<h2 style='color: blue;'>Predicted Resale Price: ${predicted_price}</h2>", unsafe_allow_html=True)