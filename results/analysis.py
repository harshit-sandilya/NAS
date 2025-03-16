import matplotlib.pyplot as plt
import re

output_files = [f"./logs/output{i+1}.log" for i in range(10)]
print(output_files)

joint_text = ""
for file_name in output_files:
    with open(file_name, 'r') as file:
        joint_text += file.read()
print(joint_text)

pattern = r"=====================================(.*?)====================================="
extracted_data = re.findall(pattern, joint_text, re.DOTALL)
print(extracted_data)

def parse_string_to_dict(input_string):
    result = {}
    lines = input_string.strip().split('\n')

    for line in lines:
        if "===>" in line:
            parts = line.split("===>")
            key = parts[0].strip()
            value = parts[1].strip()

            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            result[key] = value

    return result

result = [parse_string_to_dict(data) for data in extracted_data]
print(result)

def find_continuous_sequences(arr):
    result = []
    i = 0

    while i < len(arr):
        if arr[i] == 0 and i + 23 < len(arr):
            is_continuous = True
            for j in range(1, 24):
                if arr[i + j] != j:
                    is_continuous = False
                    break

            if is_continuous:
                for j in range(24):
                    result.append(i + j)
                i += 24
            else:
                i += 1
        else:
            i += 1
    return result

property = 'REWARD'
steps = [data['STEP'] for data in result]
indices = find_continuous_sequences(steps)
rewards = [data[property] for data in result]

rewards = [rewards[i] for i in indices]
print(rewards)

starting_indices = list(range(0, len(rewards), 24))
print(starting_indices)

split_rewards = [rewards[start: start + 24] for start in starting_indices]
split_rewards = [sequence for sequence in split_rewards if sequence == split_rewards[0] or sequence == split_rewards[-1]]
print(split_rewards)

plt.figure(figsize=(20, 12))
x_values = list(range(24))

for i, sequence in enumerate(split_rewards):
    if len(sequence) != 24:
        continue
    color = plt.cm.jet(i / len(split_rewards))
    plt.plot(x_values, sequence, label=f'Sequence {i+1}', color=color, alpha=0.7, linewidth=2)
    plt.scatter(x_values, sequence, color=color, s=30, zorder=3)

if len(split_rewards) > 15:
    plt.legend(fontsize='small', ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.05))
else:
    plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)

plt.xlabel('Step within Sequence (0-23)')
plt.ylabel(property.title())
plt.title(f'{property.title()} for Each Sequence (0-23 scale)')

plt.xticks(range(0, 24, 2))
plt.tight_layout()
plt.savefig(f'./two/{property.title()}.png', dpi=300)

# plt.figure(figsize=(20, 12))
# plt.plot(range(len(rewards)), rewards, color='blue', zorder=2)
# plt.scatter(range(len(rewards)), rewards, color='green', s=10, zorder=3)
# for start_index in starting_indices:
#     plt.axvline(x=start_index, color='red', linestyle='--', zorder=1)
# plt.xlabel('Steps')
# plt.ylabel(property.title())
# plt.title(f'{property.title()} vs Steps')
# plt.savefig(f'aggerate/{property.title()}.png')
