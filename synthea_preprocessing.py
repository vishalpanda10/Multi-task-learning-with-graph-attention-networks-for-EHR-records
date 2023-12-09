#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import json
import tqdm
import numpy as np
import base64


# In[2]:


get_ipython().system('pwd')


# In[3]:


#os.listdir('synthea/output/fhir/')


# In[4]:


directory='synthea/synthea_raw_data/fhir/'
list_files=os.listdir(directory)


# In[ ]:





# In[5]:


resource_type = []
resource_details = []
found = False

for file in range(100):
    file_path = os.path.join(directory, list_files[file])
    patient = json.load(open(file_path, 'r'))
    for resource in patient['entry']:
        resource_type.append(resource['resource']['resourceType'])
        if resource['resource']['resourceType'] == 'Medication':
            resource_details.append(resource['resource'])
            print(f"First occurrence of 'Medication' found in file: {list_files[file]}")
            found = True
            break
            
    if found:
        break


# In[ ]:





# In[8]:


set(resource_type)


# In[9]:


resource_details


# In[ ]:





# In[2]:


def extract_patient_info(file):
    patient=json.load(open(file,'r'))
    patient_df=pd.json_normalize(patient['entry'])
    observations=[]
    Immunization=[]
    Imaging=[]
    AllergyIntolerance=[]
    is_first_encounter = True
    encounter_data={}
    patient_name=None
    all_encounters=[]
    for idx,resource in enumerate(list(patient_df['resource.resourceType'])):
        if resource=='Patient':
            given_name=' '.join(patient_df['resource.name'][idx][0]['given'])
            encounter_data['Patient_Name']=given_name
            patient_name=given_name

        if resource == 'Encounter' and encounter_data:
            if not is_first_encounter:
                encounter_data['Patient_Name']=patient_name
                encounter_data['Observations'] = observations
                encounter_data['Immunization'] = Immunization
                encounter_data['ImagingStudy'] = Imaging
                encounter_data['AllergyIntolerance'] = AllergyIntolerance
                all_encounters.append(encounter_data)
                encounter_data = {}
                observations = []
                Immunization = []
                Imaging = []
                AllergyIntolerance = []
        
            encounter_data['Encounter_Type'] = patient_df['resource.type'][idx][0]['text']
            encounter_data['Encounter_Start'] = patient_df['resource.period.start'][idx]
            encounter_data['Encounter_End'] = patient_df['resource.period.end'][idx]
            is_first_encounter = False        
        
        
        elif resource =='AllergyIntolerance':
            AllergyIntolerance.append(patient_df['resource.code.coding'][idx][0]['display'])
            
        
        elif resource=='Condition':
             encounter_data['Condition']=patient_df['resource.code.text'][idx]

                
        elif resource=='Observation':
            observation_dict = {
                 patient_df['resource.code.coding'][idx][0]['display']: patient_df['resource.valueQuantity.value'][idx]
            }
            observations.append(observation_dict)
        
    

        elif resource=='Procedure':
            encounter_data['Procedure']=patient_df['resource.code.coding'][idx][0]['display']
            
        
        elif resource == 'MedicationRequest':   
            encounter_data['Medication_Request']=patient_df['resource.medicationCodeableConcept.text'][idx]
            
            if 'resource.reasonReference' in patient_df and isinstance(patient_df['resource.reasonReference'][idx], list):
                encounter_data['Medication_Request_Reason'] = patient_df['resource.reasonReference'][idx][0]['display']
            else:
                encounter_data['Medication_Request_Reason'] = None
            
        
        elif resource == 'MedicationAdministration':   
            encounter_data['Medication_Administration']=patient_df['resource.medicationCodeableConcept.text'][idx]
            
            if 'resource.reasonReference' in patient_df and isinstance(patient_df['resource.reasonReference'][idx], list):
                encounter_data['Medication_Administration_Reason']=patient_df['resource.reasonReference'][idx][0]['display']
            else:
                encounter_data['Medication_Request_Reason'] = None
        
        elif resource == 'Medication':   
            encounter_data['Medication']=patient_df['resource.code.coding'][idx][0]['display']

        
        elif resource == 'Immunization':   
            Immunization.append(patient_df['resource.vaccineCode.text'][idx])
    

        
        elif resource=='DiagnosticReport':
            patient_df['resource.code.coding'][idx][0]['display'] 
        
            presented_form = patient_df['resource.presentedForm'][idx] 
            if not pd.isna(presented_form) and isinstance(presented_form, list) and len(presented_form) > 0:
                data = presented_form[0].get('data', None)
                if data:
                    encounter_data['DiagnosticReport_Data']=base64.b64decode(data)

        elif resource == 'DocumentReference':
            document_form = patient_df['resource.content'][idx] 
            if not pd.isna(document_form) and isinstance(document_form, list) and len(document_form) > 0:
                attachment = document_form[0].get('attachment', None)
                if attachment:
                    data = attachment.get('data', None)
                    if data:
                        encounter_data['DocumentReference_Data']=base64.b64decode(data)

                
                  
        
        elif resource =='ImagingStudy':
            Imaging.append(patient_df['resource.procedureCode'][idx][0]['text'])
            Imaging.append(patient_df['resource.series'][idx][0]['modality']['display'])
            Imaging.append(patient_df['resource.series'][idx][0]['instance'][0]['title'])   

        
        
    if encounter_data:
        encounter_data['Patient_Name']=patient_name
        encounter_data['Observations'] = observations
        encounter_data['Immunization'] = Immunization
        encounter_data['ImagingStudy'] = Imaging
        encounter_data['AllergyIntolerance'] = AllergyIntolerance
        all_encounters.append(encounter_data)

    columns_order = [
    'Patient_Name',
    'Encounter_Type',
    'Encounter_Start',
    'Encounter_End',
    'Condition',
    'Observations',
    'AllergyIntolerance',
    'Procedure',
    'Medication_Request',
    'Medication_Request_Reason',
    'Medication_Administration',
    'Medication_Administration_Reason',
    'Medication',
    'Immunization',
    'DiagnosticReport_Data',
    'DocumentReference_Data',
    'ImagingStudy']
    
    patient_csv = pd.DataFrame(all_encounters).reindex(columns=columns_order)
    
    return patient_csv


# In[16]:


if not os.path.exists('Patient_Data'):
    os.makedirs('Patient_Data')
for file in range(3000):
    file_path = os.path.join(directory, list_files[file])
    patient_data=extract_patient_info(file_path)
    output_path = os.path.join('Patient_Data', list_files[file].split('-')[0] + '.csv')
    patient_data.to_csv(output_path,index=False)


# In[4]:


directory='synthea/synthea_raw_data_inference/fhir/'
list_files=os.listdir(directory)
if not os.path.exists('Patient_Data_Inference'):
    os.makedirs('Patient_Data_Inference')
for file in range(998):
    file_path = os.path.join(directory, list_files[file])
    patient_data=extract_patient_info(file_path)
    output_path = os.path.join('Patient_Data_Inference', list_files[file].split('-')[0] + '.csv')
    patient_data.to_csv(output_path,index=False)


# In[2]:


aaron=pd.read_csv('Patient_Data/Abel832_Jacobi462_ef96062b.csv')
aaron['DiagnosticReport_Data'] = aaron['DiagnosticReport_Data'].apply(lambda x: str(x).replace('\\n', ' '))
aaron['DocumentReference_Data'] = aaron['DocumentReference_Data'].apply(lambda x: str(x).replace('\\n', ' '))

aaron.head()


# In[ ]:





# In[55]:


str(list(aaron['DiagnosticReport_Data'])[1]).replace('\\n',' ')


# In[63]:


dict(aaron.iloc[25,:])


# In[ ]:


{'Unnamed: 0': 25,
 'Patient_Name': 'Abel832 Danny396',
 'Encounter_Type': 'General examination of patient (procedure)',
 'AllergyIntolerance': '[]',
 'Encounter_Start': '1985-08-28T20:24:44-04:00',
 'Encounter_End': '1985-08-28T21:10:14-04:00',
 'Condition': 'Stress (finding)',
 'Procedure': nan,
 'Medication_Request': 'amLODIPine 2.5 MG Oral Tablet',
 'Medication_Request_Reason': 'Essential hypertension (disorder)',
 'Medication_Administration': nan,
 'Medication_Administration_Reason': nan,
 'Medication': nan,
 'Observations': '[]',
 'Immunization': '[]',
 'DiagnosticReport_Data': "b' 1985-08-28  # Chief Complaint - Fatigue - Thirst - Frequent Urination - Hunger   # History of Present Illness Abel832  is a 29 year-old nonhispanic white male. Patient has a history of risk activity involvement (finding), medication review due (situation), stress (finding).  # Social History Patient is single. Patient has never smoked.  Patient identifies as heterosexual.  Patient comes from a middle socioeconomic background.  Patient did not finish high school. Patient currently has NO INSURANCE.  # Allergies No Known Allergies.  # Medications lisinopril 10 mg oral tablet; hydrochlorothiazide 25 mg oral tablet; amlodipine 2.5 mg oral tablet  # Assessment and Plan Patient is presenting with medication review due (situation), chronic kidney disease stage 1 (disorder), disorder of kidney due to diabetes mellitus (disorder), stress (finding).   ## Plan  The patient was prescribed the following medications: - hydrochlorothiazide 25 mg oral tablet - lisinopril 10 mg oral tablet - amlodipine 2.5 mg oral tablet '",
 'DocumentReference_Data': "b' 1985-08-28  # Chief Complaint - Fatigue - Thirst - Frequent Urination - Hunger   # History of Present Illness Abel832  is a 29 year-old nonhispanic white male. Patient has a history of risk activity involvement (finding), medication review due (situation), stress (finding).  # Social History Patient is single. Patient has never smoked.  Patient identifies as heterosexual.  Patient comes from a middle socioeconomic background.  Patient did not finish high school. Patient currently has NO INSURANCE.  # Allergies No Known Allergies.  # Medications lisinopril 10 mg oral tablet; hydrochlorothiazide 25 mg oral tablet; amlodipine 2.5 mg oral tablet  # Assessment and Plan Patient is presenting with medication review due (situation), chronic kidney disease stage 1 (disorder), disorder of kidney due to diabetes mellitus (disorder), stress (finding).   ## Plan  The patient was prescribed the following medications: - hydrochlorothiazide 25 mg oral tablet - lisinopril 10 mg oral tablet - amlodipine 2.5 mg oral tablet '",
 'ImagingStudy': '[]'}


