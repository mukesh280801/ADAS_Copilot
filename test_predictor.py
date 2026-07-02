from prediction.predictor import predict_future

trajectory = [
    [320, 200],
    [322, 201],
    [324, 202],
    [326, 203],
    [328, 204],
    [330, 205],
    [332, 206],
    [334, 207],
    [336, 208],
    [338, 209]
]

future = predict_future(trajectory)

print("Future Trajectory:")
print(future)