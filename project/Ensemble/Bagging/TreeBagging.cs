using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Lang;
using Solvers;

namespace Ensemble.Bagging
{
    /// <summary>
    /// IEnumerable&lt;DDataRecord&lt;string&gt;&gt; training_sample = LoadTrainingSamples();
    /// IEnumerable&lt;DDataRecord&lt;string&gt;&gt; testing_sample = LoadTestingSamples();
    /// 
    /// TreeBagging&lt;DDataRecord, string&gt; classifier = new TreeBagging&lt;DDataRecord, string&gt;(
    /// (t)=>
    /// {
    ///   //create and return a classifier such as a decision tree or perceptron
    /// }, 900, 2.0 / 3);
    /// classifier.Train(training_sample);
    /// 
    /// foreach(DDataRecord rec in testing_sample)
    /// {
    ///     string predicted_label = classifier.Predict(rec);
    /// }
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="U"></typeparam>
    public class TreeBagging<T, U> : Classifier<T, U>
        where T : DataRecord<U>
    {
        protected double mPercentageDataUsage = 0.6667;
        protected Classifier<T, U>[] mClassifiers = null;
        protected int mForestSize;

        public delegate Classifier<T, U> ClassifierGenerationMethod(int i);

        public TreeBagging(ClassifierGenerationMethod generator, int forest_size=800, double percentage_data_use=0.6667)
        {
            mForestSize = forest_size;
            mPercentageDataUsage = percentage_data_use;
            mClassifiers = new Classifier<T, U>[forest_size];
            for(int t=0; t < forest_size; ++t)
            {
                mClassifiers[t]=generator(t);
            }
        }

        public override void Train(IEnumerable<T> data_store)
        {
            List<T> temp_samples = new List<T>();
            foreach (T rec in data_store)
            {
                temp_samples.Add(rec);
            }
            int sample_count = (int)(temp_samples.Count * mPercentageDataUsage);

            for (int t = 0; t < mForestSize; ++t)
            {
                List<T> new_training_sample = new List<T>();
                for (int i = 0; i < sample_count; ++i)
                {
                    int sample_index = RandomEngine.NextInt(sample_count);
                    new_training_sample.Add(temp_samples[sample_index]);
                }
                mClassifiers[t].Train(new_training_sample);
            }
        }

        public override string Predict(T rec)
        {
            Dictionary<string, int> votes = new Dictionary<string, int>();
            foreach (Classifier<T, U> classifier in mClassifiers)
            {
                string predicted_class_variable_value = classifier.Predict(rec);
                if (votes.ContainsKey(predicted_class_variable_value))
                {
                    votes[predicted_class_variable_value]++;
                }
                else
                {
                    votes[predicted_class_variable_value] = 1;
                }
            }

            int highest_vote_count = 0;
            string highest_vote = null;
            foreach (string predicted_class_variable_value in votes.Keys)
            {
                int vote_count = votes[predicted_class_variable_value];
                if (highest_vote_count < vote_count)
                {
                    highest_vote_count = vote_count;
                    highest_vote = predicted_class_variable_value;
                }
            }

            return highest_vote;
        }

    }
}
