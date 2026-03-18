import traceback

try:
    print("Attempting to load recommender module...")
    import recommender
    print("Models loaded successfully!")
except Exception as e:
    print("Error during loading:")
    traceback.print_exc()
