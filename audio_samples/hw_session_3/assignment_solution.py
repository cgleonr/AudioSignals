import numpy as np
import hmm  # This imports the hmm.py file you already have

# ==========================================
# TASK 2: EXPLAIN A, B, PI AND RUN DECODER
# ==========================================
def task_2_mood_detector():
    print("\n" + "="*40)
    print("TASK 2: The 'Mood Detector' Model")
    print("="*40)

    # --- 1. Define the Model (The Logic) ---
    # Scenario: 
    # Hidden States (Mood): 0 = Sad, 1 = Happy
    # Observations (Food):  0 = IceCream, 1 = Salad

    # Pi: Start Probability
    # Explanation: We assume 80% chance you start the day Happy.
    pi = np.array([0.2, 0.8])

    # A: Transition Matrix (State to State)
    # Row 0 (Sad): 70% stay Sad, 30% become Happy
    # Row 1 (Happy): 40% become Sad, 60% stay Happy
    A = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])

    # B: Emission Matrix (State to Symbol)
    # Row 0 (Sad): 90% eat IceCream, 10% eat Salad
    # Row 1 (Happy): 30% eat IceCream, 70% eat Salad
    B = np.array([
        [0.9, 0.1],
        [0.3, 0.7]
    ])

    print("--- Model Parameters ---")
    print(f"Pi (Start):\n{pi}")
    print(f"A (Transition - Mood Swing):\n{A}")
    print(f"B (Emission - Food Choice):\n{B}")

    # --- 2. Define Observations ---
    # We see someone eat: IceCream, IceCream, Salad, Salad, IceCream
    observations = np.array([0, 0, 1, 1, 0])
    
    # --- 3. Run Viterbi Decoder ---
    hidden_states = hmm.viterbi(observations, A, B, pi)

    # --- 4. Analyse Outcome ---
    print(f"\n--- Decoding Results ---")
    print(f"Observations (Food): {observations}")
    print(f"Decoded States (Mood): {hidden_states}")
    
    # Text interpretation
    moods = ["Sad", "Happy"]
    foods = ["IceCream", "Salad"]
    print("\nInterpreted Path:")
    for obs, state in zip(observations, hidden_states):
        print(f"  Ate {foods[obs]} -> Must be {moods[state]}")

# ==========================================
# TASK 3: ADD NEW SYMBOL '!' MANUALLY
# ==========================================
def task_3_add_symbol():
    print("\n" + "="*40)
    print("TASK 3: Adding Symbol '!' (Index 2)")
    print("="*40)

    # We take the A and Pi from Task 2, but we must expand B.
    # Old B was (2, 2). New B must be (2, 3) to accommodate symbol '2' (!).
    
    pi = np.array([0.2, 0.8])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])

    # New Scenario: '!' represents "Spilling Food".
    # Sad people spill food more often because they are crying.
    # Happy people spill less.
    
    # We must ensure rows sum to 1.0.
    # Sad:   IceCream(0.8), Salad(0.1), Spill(0.1)
    # Happy: IceCream(0.3), Salad(0.65), Spill(0.05)
    
    B_new = np.array([
        [0.80, 0.10, 0.10], 
        [0.30, 0.65, 0.05] 
    ])

    print("--- New B Matrix (Manual Expansion) ---")
    print(B_new)

    # Test with the new symbol '2' (!)
    # Obs: IceCream(0), Spill(2), Salad(1)
    observations = np.array([0, 2, 1])
    
    hidden_states = hmm.viterbi(observations, A, B_new, pi)
    
    print(f"\nTest Observation with '!': {observations}")
    print(f"Decoded States: {hidden_states}")
    print("(Note: If the model sees '!' (2), it likely calculates 'Sad' (0) because Sad has higher emission prob for '!'.)")

# ==========================================
# TASK 4: ENFORCE START AND STOP STATES
# ==========================================
def task_4_start_stop():
    print("\n" + "="*40)
    print("TASK 4: Enforcing Start and Stop")
    print("="*40)

    # 1. Enforce Start State
    # We force the model to ALWAYS start at State 0 (Sad).
    pi_forced = np.array([1.0, 0.0])

    # 2. Enforce Stop State
    # We treat State 1 (Happy) as a "Trap" or "Absorbing State".
    # Once you enter State 1, you can NEVER leave (Probability 1.0 to stay).
    # State 0 can go to 1, but 1 cannot go to 0.
    
    A_forced = np.array([
        [0.5, 0.5],  # 50% chance to stay Sad, 50% chance to reach Nirvana (Stop)
        [0.0, 1.0]   # 0% chance to get Sad again. 100% stay Happy/Stop.
    ])

    B_forced = np.array([
        [0.9, 0.1], 
        [0.1, 0.9]
    ])

    print("--- Enforced Matrices ---")
    print(f"Pi (Always Start Sad): {pi_forced}")
    print(f"A (State 1 is a Trap/Stop):\n{A_forced}")

    # Test: Even if we see "IceCream" (Sad food) late in the sequence, 
    # the model should refuse to switch back to Sad because it's trapped.
    obs = np.array([0, 1, 1, 0, 0]) # Ice, Salad, Salad, Ice, Ice
    path = hmm.viterbi(obs, A_forced, B_forced, pi_forced)
    
    print(f"\nObservation: {obs}")
    print(f"Path:        {path}")
    print("Notice: Once it switches to 1, it stays 1, even though observations (0) suggest state 0.")

# ==========================================
# TASK 5: PARAMETER INVESTIGATION
# ==========================================
def task_5_investigate_params():
    print("\n" + "="*40)
    print("TASK 5: Effect of Training Iterations")
    print("="*40)

    # We will try to 'learn' the Mood model from data.
    # True Model (Target):
    A_true = np.array([[0.7, 0.3], [0.4, 0.6]])
    B_true = np.array([[0.9, 0.1], [0.3, 0.7]])
    pi_true = np.array([0.5, 0.5])
    
    # Generate Synthetic Data from True Model
    # (Hardcoded for consistency in this example)
    # A long sequence allows better learning
    observations = np.array([0,0,0,1,1,1,1,0,0,1,1,0,0,0,0,1] * 5) 

    # Initial Random Guess
    np.random.seed(42)
    A_init, B_init, pi_init = hmm.new_model(num_states=2, num_symbols=2)
    
    print("--- Experiment 1: Low Training (1 Iteration) ---")
    # Using Baum-Welch from hmm.py
    A_1, B_1 = hmm.baum_welch(observations, A_init, B_init, pi_init, n_iter=1)
    print("Learned A (1 iter):\n", np.round(A_1, 2))
    print("Note: Values likely far from True A (0.7, 0.3 / 0.4, 0.6)")

    print("\n--- Experiment 2: High Training (100 Iterations) ---")
    A_100, B_100 = hmm.baum_welch(observations, A_init, B_init, pi_init, n_iter=100)
    print("Learned A (100 iter):\n", np.round(A_100, 2))
    print("Note: Values should be closer to True A, proving iterations improve accuracy.")
    print("(Note: HMMs can flip states, so State 0 might become State 1. Look for the pattern, not just indices.)")

if __name__ == "__main__":
    task_2_mood_detector()
    task_3_add_symbol()
    task_4_start_stop()
    task_5_investigate_params()