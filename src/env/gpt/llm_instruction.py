import openai   
import ast     
import torch 
import os


def gpt4_instruct(real_points, danger_points, device):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    prompt = "You are now a route planning assistant in a three-dimensional space. You are given a series of three-dimensional coordinates [x, y, z], which include points that must be passed through in order and points that must be avoided. The points that must be passed through in order are enclosed in square brackets, and the points that must be avoided are enclosed in another set of square brackets. For example, if you need to pass through [0.0,0.0,1.0], [1.0,1.0,2.0] and avoid [0.5,0.5,1.5], the user would input [[[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], [[0.5, 0.5, 1.5]]]. You need to provide a new set of points that must be passed through in order. When passing through these points in order, the distance to all points to be avoided should be greater than 0.2. Also, you need to indicate the indices of the points you added. The new ordered points are enclosed in square brackets, and the indices are enclosed in another set of square brackets. Please do not output any other content. For example, if the new sequence is [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 2.0]], and the new points added are the first and third points (zero-based), then your output should be [[[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 2.0]], [1, 2]]."

    resp = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "[[[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], [[0.5, 0.5, 1.5]]]"},
            {
                "role": "assistant",
                "content": "[[[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 2.0]], [1, 2]]",
            },
            {"role": "user", "content": str([real_points.tolist(), danger_points.tolist()])},
        ],
        temperature=0.3,
    )

    resp_list = ast.literal_eval(resp['choices'][0]['message']['content'])
    print("openai reply:",resp_list)

    target_points = torch.tensor(resp_list[0]).to(device)
    pseudo_index = torch.tensor(resp_list[1]).to(device)
    return target_points, pseudo_index