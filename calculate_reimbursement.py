import joblib
import sys

# Load the trained model
model = joblib.load("rf_model.joblib")

def predict(trip_duration_days, miles_traveled, total_receipts_amount):
    """Predict reimbursement amount using the Random Forest model."""
    features = [[trip_duration_days, miles_traveled, total_receipts_amount]]
    return round(model.predict(features)[0], 2)

if __name__ == "__main__":
    # Read inputs from command line (provided by run.sh)
    if len(sys.argv) != 4:
        print("Error: Expected 3 arguments (trip_duration_days, miles_traveled, total_receipts_amount)", file=sys.stderr)
        sys.exit(1)
    
    try:
        trip_duration_days = float(sys.argv[1])
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])
        
        # Make prediction and output result
        prediction = predict(trip_duration_days, miles_traveled, total_receipts_amount)
        print(f"{prediction:.2f}")
        
    except ValueError:
        print("Error: All inputs must be valid numbers", file=sys.stderr)
        sys.exit(1)
