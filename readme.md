```bash
#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #no
from tqdm import tqdm
from scipy.special import expit

#define binary variables

def binary(odds_bin, prob0, prob1):
    variables_bin = []
    for i in range(len(odds_bin)):
        binary = np.random.choice(2, p=[prob0[i], prob1[i]])
        variables_bin.append(binary)
    return variables_bin

#define numeric variables

def numeric(odds_num, min_variable, max_variable):
    variables_num = []
    for i in range(len(odds_num)):
        min_var= min_variable[i]
        max_var = max_variable[i]
        numeric = np.random.randint(min_var, max_var)
        variables_num.append(numeric)
    return variables_num

#calculate probability

def sigmoid(base, OR_sex, sex, odds_bin, odds_num, variables_bin, variables_num):
    log_odds = np.log(base)
    log_odds += np.log(OR_sex)*sex
    for i, var in enumerate(variables_bin):
        log_odds += np.log(odds_bin[i])*var
    for i, var in enumerate(variables_num):
        log_odds += np.log(odds_num[i])*var
    prob = 1.0/(1.0+np.exp(-log_odds))
    return prob

#decison making

def decision(P): 
    return 1 if P > np.random.random() else 0

#make age ranges
def classify_age(age):
    if age < 18:
        return 'under 18'
    elif age < 30:
        return '18-29'
    elif age < 45:
        return '30-44'
    elif age < 60:
        return '45-59'
    else:
        return '60 and above'
    
# Label each bar with the count
def autolabel(ax, groups):
    for group in groups:
        height = group.get_height()
        ax.annotate('{}'.format(height),
                    xy=(group.get_x() + group.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


#simulate data

def simulate(base, OR_sex, odds_bin, prob0, prob1, odds_num, min_variable, max_variable, n_samples, col_names):
    np.random.seed(0)
    counts = {"zerosw": 0, "zerosm": 0, "onesw": 0, "onesm": 0}
    counts_age_non_suicidal = {"18-29": 0, "30-44": 0, "45-59": 0, "60 and above": 0}
    counts_age_suicidal = {"under 18": 0, "18-29": 0, "30-44": 0, "45-59": 0, "60 and above": 0}
    dataset = []
    
    #decide sex

    for i in tqdm(range(n_samples), colour= "green", desc="Loading..."):

        if i < n_samples/2:
            sex = 0
            real = 'woman'
        else:
            sex = 1
            real = 'man'

        #obtain the output of a random input
          
        variables_bin = binary(odds_bin, prob0, prob1)
        variables_num = numeric(odds_num, min_variable, max_variable)
        output_y = decision(sigmoid(base, OR_sex, sex, odds_bin, odds_num, variables_bin, variables_num))

        #build the dataset
        
        dataset.append("sex": sex, 
                        **{col_names[j+1]: variables_bin[j] for j in range(len(variables_bin))}, 
                        **{col_names[k+(len(variables_bin)+1)]: variables_num[k] for k in range(len(variables_num))}, 
                        "Output": output_y)
        
        
        #store the results for the graphs

        if sex == 0: 
            if output_y == 0:
                counts["zerosw"] = counts["zerosw"] + 1
            else:
                counts["onesw"] = counts["onesw"] + 1
        if sex == 1:
            if output_y == 0:
                counts["zerosm"] = counts["zerosm"] + 1
            else:
                counts["onesm"] = counts["onesm"] + 1

        age_range = classify_age(variables_num[0])
        if output_y == 0:
            counts_age_non_suicidal[age_range] += 1
        else:
            counts_age_suicidal[age_range] +=1
      
    dataset_df = pd.DataFrame(dataset)
    dataset_df.to_csv('dataset.csv', index=False)

    #dataset by sex
           
    databysex = pd.DataFrame({'gender': ['women', 'men'], 'non_suicidal': 
                                   [round((counts["zerosw"]/(n_samples*0.5))*100), round(counts["zerosm"]/(n_samples*0.5)*100)], 
                                   'suicidal': [round((counts["onesw"]/(n_samples*0.5))*100), round((counts["onesm"]/(n_samples*0.5))*100)]})
    
    #dataset by age

    databyage = pd.DataFrame({'Age': ["18-29", "30-44", "45-59", "60 and above"], 'non_suicidal': 
                                   [round((counts_age_non_suicidal["18-29"]/(counts_age_non_suicidal["18-29"]+counts_age_suicidal["18-29"]))*100), 
                                    round((counts_age_non_suicidal["30-44"]/(counts_age_non_suicidal["30-44"]+counts_age_suicidal["30-44"]))*100),
                                    round((counts_age_non_suicidal["45-59"]/(counts_age_non_suicidal["45-59"]+counts_age_suicidal["45-59"]))*100),
                                    round((counts_age_non_suicidal["60 and above"]/(counts_age_non_suicidal["60 and above"]+counts_age_suicidal["60 and above"]))*100)],
                                   'suicidal': 
                                   [round((counts_age_suicidal["18-29"]/(counts_age_non_suicidal["18-29"]+counts_age_suicidal["18-29"]))*100), 
                                    round((counts_age_suicidal["30-44"]/(counts_age_non_suicidal["30-44"]+counts_age_suicidal["30-44"]))*100),
                                    round((counts_age_suicidal["45-59"]/(counts_age_non_suicidal["45-59"]+counts_age_suicidal["45-59"]))*100),
                                    round((counts_age_suicidal["60 and above"]/(counts_age_non_suicidal["60 and above"]+counts_age_suicidal["60 and above"]))*100)]})

    #plot

    x1 = np.arange(len(databysex['gender']))  # the label locations
    x2 = np.arange(len(databyage['Age']))  # the label locations


    width = 0.25  # the width of the bars

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, layout='constrained')
    fig.suptitle('Proportions of suicide tendencies', fontweight="bold", fontsize=20)

    non_suicidal1 = ax1.bar(x1 - width/2, databysex['non_suicidal'], width, label='Non-suicidal')
    suicidal1 = ax1.bar(x1 + width/2, databysex['suicidal'], width, label='Suicidal')

    # Add labels and title

    ax1.set_xlabel('Gender')
    ax1.set_ylabel('Percentage')
    ax1.set_ylim(0, 100)
    ax1.set_title('Proportions by Gender')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(databysex['gender'])
    ax1.legend()
    autolabel(ax1, non_suicidal1)
    autolabel(ax1, suicidal1)

    non_suicidal2 = ax2.bar(x2 - width/2, databyage['non_suicidal'], width, label='Non-suicidal')
    suicidal2 = ax2.bar(x2 + width/2, databyage['suicidal'], width, label='Suicidal')

    ax2.set_xlabel('Age')
    ax2.set_ylabel('Percentage')
    ax2.set_ylim(0, 100)
    ax2.set_title('Proportions by Age')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(databyage['Age'])
    ax2.legend()
    autolabel(ax2, non_suicidal2)
    autolabel(ax2, suicidal2)

    fig.set_figheight(8)
    fig.set_figwidth(12)

    plt.show()

    return dataset_df


base = 1
OR_sex = 1.5 #sex
odds_bin = [1.2] #divorce
prob0= [0.85] #define the probability of the values for the binary variables
prob1= [0.15]
odds_num = [0.98, 1] #age, income (per year), psychiatric diagnosis (Non, anxiety disorders, psychotic disorders, substance abuse disorders, mood disorders)
min_variable = [18, 25]
max_variable = [80, 40]
n_samples =100000
col_names=['sex', 'divorce', 'age', 'income']

simulate(base, OR_sex, odds_bin, prob0, prob1, odds_num, min_variable, max_variable, n_samples, col_names)

```