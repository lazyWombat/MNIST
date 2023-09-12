Console.WriteLine("Reading data...");

using var reader = new StreamReader("train.csv");
_ = reader.ReadLine();

List<int> devLabels = new();
List<float> devData = new();
List<int> trainLabels = new();
List<float> trainData = new();

while(reader.Peek() >= 0 && devLabels.Count < 1000)
{
    var s = reader.ReadLine();    
    var split = s!.Split(',');
    devLabels.Add(int.Parse(split[0]));
    foreach(var item in split[1..])
    {
        devData.Add(float.Parse(item) / 255);
    }
}

Console.WriteLine("Got dev data");

while(reader.Peek() >= 0)
{
    var s = reader.ReadLine();    
    var split = s!.Split(',');
    trainLabels.Add(int.Parse(split[0]));
    foreach(var item in split[1..])
    {
        trainData.Add(float.Parse(item) / 255);
    }
}

Console.WriteLine("Got train data");

Network init()
{
    var w1 = new SmartArray(10, 784).Rand() - 0.5f;
    var b1 = new SmartArray(10, 1).Rand() - 0.5f;
    var w2 = new SmartArray(10, 10).Rand() - 0.5f;
    var b2 = new SmartArray(10, 1).Rand() - 0.5f;
    return new Network(w1, b1, w2, b2);
}

SmartArray PrepareLabels(List<int> labels)
{
    var result = new SmartArray(labels.Count, 10);
    for(var i = 0; i < labels.Count; i++)
    {
        result.SetValue(i, labels[i], 1);
    }
    return result.Transpose();
}

Func<float, float> reLU = x => x > 0 ? x : 0;
Func<float, float> derivation_ReLU = x => x > 0 ? 1 : 0;

SmartArray softmax(SmartArray input)
{
    var exp = input.Transpose().Apply(x => (float)Math.Pow(Math.E, x));
    var sum = exp.SumRows();
    return SmartArray.ApplyTwo(exp, sum, (a, b) => a / b).Transpose();
}

Network forward(Network net, SmartArray data)
{
    var z1 = net.w1.Dot(data) + net.b1;
    var a1 = z1.Apply(reLU);
    var z2 = net.w2.Dot(a1) + net.b2;
    var a2 = softmax(z2);
    return new Network(z1, a1, z2, a2);
}

Network backward(Network z, Network n, SmartArray data, SmartArray labels)
{
    float length = labels.Width;
    var diff_z_2 = z.b2 - labels;
    var diff_w_2 = diff_z_2.Dot(z.b1.Transpose()) / length;    
    var diff_b_2 = diff_z_2.SumRows() / length;
    var diff_z_1 = n.w2.Transpose().Dot(diff_z_2) * z.w1.Apply(derivation_ReLU);
    var diff_w_1 = diff_z_1.Dot(data.Transpose()) / length;
    var diff_b_1 = diff_z_1.SumRows() / length;
    return new Network(diff_w_1, diff_b_1, diff_w_2, diff_b_2);
}

Network update(Network network, Network diff, float learningRate) =>
    new Network(
        network.w1 - diff.w1 * learningRate,
        network.b1 - diff.b1 * learningRate,
        network.w2 - diff.w2 * learningRate,
        network.b2 - diff.b2 * learningRate);

float getAccuracy(SmartArray results, List<int> labels)
{
    var correct = 0;
    for(var i = 0; i < results.Width; i++)
    {
        var index = 0;
        var prob = results.GetValue(0, i);
        for(var j = 1; j < results.Length; j++)    
        {
            var newProb = results.GetValue(j, i);
            if (newProb > prob)
            {
                prob = newProb;
                index = j;
            }
        }
        if (index == labels[i])
        {
            correct++;
        }    
    }
    return ((float)correct) / labels.Count;
}

Network gradientDescent(SmartArray data, List<int> labels, float learningRate, int iterations)
{
    // init
    Console.WriteLine("Initializing...");
    var network = init();

    var preparedLabels = PrepareLabels(labels);

    Console.WriteLine("Iterating...");
    for(var i = 0; i < iterations; i++)
    {
        var z = forward(network, data);
        var d = backward(z, network, data, preparedLabels);
        network = update(network, d, learningRate);
        if (i % 10 == 0)
        {
            Console.WriteLine($"Iteration: {i}\tAccuracy: {getAccuracy(z.b2, labels)}");
        }
    }

    return network;
}

var trainedNetwork = gradientDescent(new SmartArray(trainData.ToArray(), 784).Transpose(), trainLabels, 0.1f, 500);
