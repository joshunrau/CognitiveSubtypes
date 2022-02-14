from dataset import Dataset

data = Dataset()
data.recode_diagnoses()
data.apply_inclusion_criteria(method="OR")
data.apply_exclusion_criteria()
data.recode_vars()
data.create_binary_variables("medication", {"taking_painkiller": "(aspirin|tylenol)"})
data.write_csv()