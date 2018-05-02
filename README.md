# cs-ensembles

Ensembles method implemented in C#

# Install

```bash
Install-Package cs-ensembles
```

# Usage

The sample codes below show how to use the AdaBoosting classifier:

```cs 
IEnumerable<DDataRecord<string>> training_sample = LoadTrainingSamples();
IEnumerable<DDataRecord<string>> testing_sample = LoadTestingSamples();

AdaBoost<DDataRecord, string> classifier = new AdaBoost<DDataRecord, string>();
classifier.CreateAndTrainWeakClassifiers(training_sample, (t)=>
{
 //create and return a weak classifier such as a decision tree or perceptron
});
classifier.Train(training_sample);

foreach(DDataRecord rec in testing_sample)
{
   string predicted_label = classifier.Predict(rec);
}
```

The sample codes below show how to use the TreeBagging classifier:

```cs
IEnumerable<DDataRecord<string>> training_sample = LoadTrainingSamples();
IEnumerable<DDataRecord<string>> testing_sample = LoadTestingSamples();

TreeBagging<DDataRecord, string> classifier = new TreeBagging<DDataRecord, string>(
(t)=>
{
  //create and return a classifier such as a decision tree or perceptron
}, 900, 2.0 / 3);
classifier.Train(training_sample);

foreach(DDataRecord rec in testing_sample)
{
    string predicted_label = classifier.Predict(rec);
} 
```
