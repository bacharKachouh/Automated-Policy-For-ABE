import os
import csv

def create_patient_directory(patient_id, base_path="."):
    """
    Create a directory for a patient named 'Patient_<ID>'.

    :param patient_id: The ID of the patient.
    :param base_path: The base directory where the patient directory will be created.
    """
    directory_name = f"Patient_{patient_id}"
    directory_path = os.path.join(base_path, directory_name)
    
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory created: {directory_path}")
    except Exception as e:
        print(f"Error creating directory: {e}")

def update_mapping_table(gid, patient_id, mapping_file):
    """
    Update the mapping table with the GID and Patient ID.

    :param gid: The Global ID of the patient.
    :param patient_id: The Patient ID.
    :param mapping_file: The path to the CSV file storing the mappings.
    """
    # Check if the mapping file exists
    file_exists = os.path.isfile(mapping_file)
    
    try:
        with open(mapping_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            # Write the header only if the file is new
            if not file_exists:
                writer.writerow(["GID", "Patient_ID"])
            # Write the mapping
            writer.writerow([gid, patient_id])
        print(f"Mapping added: GID={gid}, Patient_ID={patient_id}")
    except Exception as e:
        print(f"Error updating mapping table: {e}")

if __name__ == "__main__":
    # Base path where the directories will be created
    base_directory = "./patients"
    mapping_file = "patient_mapping.csv"
    
    # Ensure the base directory exists
    os.makedirs(base_directory, exist_ok=True)
    
    while True:
        # Ask the user for GID and Patient ID
        gid = input("Enter Patient GID (or type 'exit' to quit): ").strip()
        if gid.lower() == "exit":
            print("Exiting the program.")
            break
        
        patient_id = input("Enter Patient ID: ").strip()
        if not gid or not patient_id:
            print("GID and Patient ID cannot be empty. Please try again.")
            continue
        
        # Create the patient directory
        create_patient_directory(patient_id, base_directory)
        
        # Update the mapping table
        update_mapping_table(gid, patient_id, mapping_file)

