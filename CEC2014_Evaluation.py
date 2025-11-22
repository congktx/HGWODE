import numpy as np
import opfunu.cec_based.cec2014 as CEC2014
from tqdm import tqdm
import pandas as pd
from HGWODE_Improved import HGWODE_QLEARN_OPTIMIZER

def evaluate_cec2014(dim=30, runs=51, NP=50, max_iter=3000, seed_start=0):
    func_classes = [
        # CEC2014.F12014, CEC2014.F22014, CEC2014.F32014, 
        CEC2014.F42014, 
        CEC2014.F52014, CEC2014.F62014, CEC2014.F72014, CEC2014.F82014, 
        CEC2014.F92014, CEC2014.F102014, CEC2014.F112014, CEC2014.F122014, 
        CEC2014.F132014, CEC2014.F142014, CEC2014.F152014, CEC2014.F162014, 
        CEC2014.F172014, CEC2014.F182014, CEC2014.F192014, CEC2014.F202014,
        CEC2014.F212014, CEC2014.F222014, CEC2014.F232014, CEC2014.F242014, 
        CEC2014.F252014, CEC2014.F262014, CEC2014.F272014, CEC2014.F282014, 
        CEC2014.F292014, CEC2014.F302014
    ]
    
    results = []
    
    print("="*80)
    print("HGWODE CEC2014 BENCHMARK EVALUATION")
    print("="*80)
    print(f"Dimension: {dim}")
    print(f"Population size: {NP}")
    print(f"Max iterations: {max_iter}")
    print(f"Total FEs per run: {NP * 2 * max_iter} (GWO + DE steps)")
    print(f"Independent runs: {runs}")
    print("="*80)
    
    for fid, FuncClass in enumerate(func_classes, start=1):
        print(f"\n{'='*60}")
        print(f"Function {fid}: {FuncClass.__name__}")
        print(f"{'='*60}")
        
        errors = []
        
        for run in tqdm(range(runs), desc=f"F{fid:02d}"):
            # Initialize function
            func_obj = FuncClass(ndim=dim)
            
            # Wrapper for fitness function
            def fitness(x):
                return func_obj.evaluate(x)
            
            # Run HGWODE
            best_x, best_f, history = HGWODE_QLEARN_OPTIMIZER(
                f=fitness,
                dim=dim,
                lb=func_obj.lb,
                ub=func_obj.ub,
                NP=NP,
                max_iter=max_iter,
                F=0.5,
                CR=0.9,
                seed=seed_start + run
            )
            
            # Calculate error
            f_opt = func_obj.f_global
            error = best_f - f_opt
            
            errors.append(error)
        
        # Statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        median_error = np.median(errors)
        best_error = np.min(errors)
        worst_error = np.max(errors)
        
        results.append({
            'Function': f'F{fid}',
            'Mean': mean_error,
            'Std': std_error,
            'Median': median_error,
            'Best': best_error,
            'Worst': worst_error
        })
        
        print(f"Mean error:   {mean_error:.5e}")
        print(f"Std error:    {std_error:.5e}")
        print(f"Median error: {median_error:.5e}")
    
    # Create dataframe
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('cec2014_results.csv', index=False)
    print("\n✓ Results saved to: cec2014_results.csv")
    
    return df

if __name__ == '__main__':
    results_df = evaluate_cec2014(
        dim=30,          
        runs=1,         
        NP=50,           
        max_iter=3000,   
        seed_start=42
    )
    print(results_df)