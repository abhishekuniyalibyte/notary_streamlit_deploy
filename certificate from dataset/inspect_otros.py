import json

# Load customers index
with open("customers_index.json", "r", encoding="utf-8") as f:
    customers = json.load(f)

otros_files = []

# Load certificate_types.json for comparison (optional)
with open("certificate_types.json", "r", encoding="utf-8") as f:
    cert_types = json.load(f)


# Collect keywords for known certificate types
known_type_keywords = list(cert_types.keys())
known_type_keywords.remove("otros")  # remove "otros" itself

print("\nKnown certificate keywords:")
print(known_type_keywords)


def is_certificate(filename):
    name = filename.lower()

    # If filename contains one of the known keywords → certificate
    for key in known_type_keywords:
        for part in key.split("_"):
            if part in name:
                return True

    return False


# Scan all customer folders
for customer, info in customers.items():
    for file in info["files"]["certificates"]:
        filename = file["filename"]
        
        # If it's not matching ANY known type → goes to otros
        if not is_certificate(filename):
            otros_files.append(filename)


print("\n============================")
print("FULL OTROS FILE LIST")
print("============================")
print(f"Total files: {len(otros_files)}\n")

for f in otros_files:
    print(" -", f)

print("\n============================")
