import pandas as pd

def calculate_makespan(log_path):
    """
    Reads the YAFS log and calculates the MakeSpan (time last task finished).
    """
    try:
        # Load CSV
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"Error reading log: {e}")
        return 0

    if df.empty:
        print("Log is empty.")
        return 0

    # The simulation ends when the last event is processed.
    # The 'time' column holds the timestamp of every event.
    # The maximum time recorded is effectively the Makespan.
    makespan = df["time"].max()
    
    return makespan

if __name__ == "__main__":
    ms = calculate_makespan("logs/log_test.csv")
    print(f"FINAL MAKESPAN: {ms}")