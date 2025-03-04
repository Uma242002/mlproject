import streamlit as st
import pandas as pd

st.header("Compitators")

st.write("Predictive Model Built on Below Sample Data")

data = pd.read_csv("CampaInpData.csv")
st.dataframe(data.head())

col1,col2 = st.columns(2)
with col1:
    Institute_name = st.selectbox("Enter the Institute name:",data["Institute name"].unique())
    Area = st.selectbox("Enter area:",data.Area.unique())
    City = st.selectbox("Enter City:",data.City.unique())
    Aws_and_devops= st.selectbox("Enter Aws and devops is :",data["Aws and devops"].unique())
    Java_full_stack= st.selectbox("Enter Java full stack:",data["Java full stack"].unique())
    Data_science_and_AI= st.selectbox("Enter Data science and AI:",data["Data science and AI"].unique())
    python = st.selectbox("Enter python:",data.python.unique())
    
    
with col2:
    Faculty_Expertise = st.selectbox("Enter Faculty Expertise:",data["Faculty Expertise"].unique())
    Training_Methodology=st.selectbox("Enter Training Methodology:",data["Training Methodology"].unique())
    Placement_Rate = st.selectbox("Has the Placement Rate:",data["Placement Rate"].unique())
    Average_Salary_in_lpa= st.number_input(f"Enter the Average Salary in lpa (Min {data['Average Salary in lpa'].min()} to Max {data['Average Salary in lpa'].max()})")
    Placement_Support = st.selectbox("Has the Placement Support:",data["Placement Support"].unique())
    Internship_Opportunities = st.selectbox("Enter Internship Opportunities:",data["Internship Opportunities"].unique())
    Certifications_Offered=st.selectbox("Enter Certifications_Offered:",data["Certifications Offered"].unique())
    review_out_of_5 = st.number_input(f"Enter review out of 5 (Min {data['review out of 5'].min()} to Max {data['review out of 5'].max()})")


xdata = [
    Institute_name,
    Area,
    City,
    Aws_and_devops,
    Java_full_stack,  # Removed space
    Data_science_and_AI,
    python,
    Faculty_Expertise,
    Training_Methodology,
    Placement_Rate,
    Average_Salary_in_lpa,
    Placement_Support,
    Internship_Opportunities,
    Certifications_Offered,
    review_out_of_5
]


import joblib
with open('jobsvmreg.pkl','rb') as f:
    model = joblib.load(f)
with open('label_encoder.pkl','rb') as p:
    label_encoder  = joblib.load(p)
with open('sc.pkl','rb') as s:
    scaler = joblib.load(s)

x = pd.DataFrame([xdata], columns=data.columns[0:15])

st.write("Given Input:")
st.dataframe(x)
 # Feature Engineering: Need to apply same steps done for training, while giving it to model for prediction
binary_cols = ["Aws and devops", "Java full stack", "Data science and AI", "python","Certifications Offered","Internship Opportunities"]
x[binary_cols] = x[binary_cols].apply(lambda k: k.map({"yes": 1, "no": 0}))

x.City.replace({'hyderabad':0,'banglore':1}, inplace=True)  
x['Training Methodology'].replace({'online and  offline training':0, 'offline training':1,'online training':2},inplace=True)
x["Placement Rate"].replace({"average":0, "good":1}, inplace=True)
x["Faculty Expertise"].replace({"experience":1, "well qualified":0}, inplace=True)    
def categorize_support(text):
      text = str(text).lower()  # Convert to lowercase for uniformity
      
      categories = {
          'Resume_Assistance': any(keyword in text for keyword in ['resume', 'linkedin']),
          'Interview_Training': any(keyword in text for keyword in ['interview', 'soft skills']),
          'Job_Placement': any(keyword in text for keyword in ['job placement', 'placement assistance']),
          'Project_Training': any(keyword in text for keyword in ['real time projects', 'internship', 'project'])
      }
    
      return pd.Series(categories)

     # Apply the grouping function
df_encoded = x['Placement Support'].apply(categorize_support)
x= pd.concat([x, df_encoded], axis=1)
x = x.drop(columns=['Placement Support'])
job = ['Resume_Assistance','Interview_Training', 'Job_Placement', 'Project_Training']
x[job] = x[job].astype(bool)
x[job] = x[job].apply(lambda k: k.map({True: 1, False: 0}))
# LabelEncoding
for col in ["Institute name", "Area"]:
      x[f"{col}"] = label_encoder.fit_transform(x[col])

    # Scaling
#numerical_cols = ["Institute name", "Area","Training Methodology", "Average Salary in lpa", "review out of 5"]

    # Apply scaling only to numerical columns
#x[numerical_cols] = scaler.fit_transform(x[numerical_cols])
    
    
import numpy as np
if st.button("Predict"):
    prediction = model.predict(x)  # Get model prediction

    # Check model output type
    #print(f"Model Output Type: {type(prediction)}")
    
    if isinstance(prediction, (list, tuple, np.ndarray)):
        min_fee, max_fee = prediction[0] if len(prediction[0]) == 2 else (prediction[0], None)
    #else:
       # min_fee = prediction
       # max_fee = None  # If model doesn't return a range

    # Display the results in Streamlit
    if max_fee:
        st.write(f"🎯 **Predicted Min Fee:** {round(min_fee, 2)} LPA")
        st.write(f"🎯 **Predicted Max Fee:** {round(max_fee, 2)} LPA")
    else:
        st.write(f"🎯 **Predicted Fee:** {round(min_fee, 2)} LPA")

