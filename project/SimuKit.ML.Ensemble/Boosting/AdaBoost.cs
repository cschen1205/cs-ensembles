using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SimuKit.ML.Lang;
using SimuKit.ML.Solvers;

namespace SimuKit.ML.Ensemble.Boosting
{
    /// <summary>
    /// Discrete binary classifier by default
    /// Usage:
    /// IEnumerable&lt;DDataRecord&lt;string&gt;&gt; training_sample = LoadTrainingSamples();
    /// IEnumerable&lt;DDataRecord&lt;string&gt;&gt; testing_sample = LoadTestingSamples();
    /// 
    /// AdaBoost&lt;DDataRecord, string&gt; classifier = new AdaBoost&lt;DDataRecord, string&gt;();
    /// classifier.CreateAndTrainWeakClassifiers(training_sample, (t)=>
    /// {
    ///   //create and return a weak classifier such as a decision tree or perceptron
    /// });
    /// classifier.Train(training_sample);
    /// 
    /// foreach(DDataRecord rec in testing_sample)
    /// {
    ///     string predicted_label = classifier.Predict(rec);
    /// }
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="U"></typeparam>
    public class AdaBoost<T, U> : EnsembleLearning<T, U> 
        where T : DataRecord<U>
    {
        protected string mClassVariableValue_YES;
        protected int mLayer;
        protected double[] mAlphaValues;
        protected double mThreshold = 0;
        public const string ClassVariableValue_No = "No@AdaBoost";

        public delegate WeakClassifier<T, U> WeakClassifierGenerationMethod(IEnumerable<T> data_store, int t);

        public AdaBoost(string class_variable_value_YES, int T, double threshold = 0)
        {
            mClassVariableValue_YES = class_variable_value_YES;
            mLayer = T;
            mWeakClassifiers = new WeakClassifier<T, U>[mLayer];
            mAlphaValues = new double[mLayer];
            mThreshold = threshold;
        }

        public void CreateAndTrainWeakClassifiers(IEnumerable<T> data_store, WeakClassifierGenerationMethod generator)
        {
            for (int t = 0; t < mLayer; ++t)
            {
                mWeakClassifiers[t] = generator(data_store, t);
            }
        }

        public override void Train(IEnumerable<T> data_store)
        {
            double h = 0;
            double y = 0;

            int record_count = 0;
            foreach (T rec in data_store)
            {
                record_count++;
            }
            double[] weights = new double[record_count];

            for (int i = 0; i < record_count; ++i)
            {
                weights[i] = 1.0 / record_count;
            }

            int sample_index = 0;
            double epsilon = 0;

            for (int t = 0; t < mLayer; ++t)
            {
                // Choose f_t(x)
                double min_epsilon = double.MaxValue;
                WeakClassifier<T, U> best_classifier = null;
                foreach (WeakClassifier<T, U> classifier in mWeakClassifiers)
                {
                    sample_index = 0;
                    epsilon = 0;
                    foreach (T rec in data_store)
                    {
                        h = classifier.WeakPredict(rec);
                        y = (mClassVariableValue_YES == rec.Label ? 1 : -1);

                        epsilon += weights[sample_index] * System.Math.Exp(-h * y);

                        sample_index++;
                    }
                    if (min_epsilon > epsilon)
                    {
                        min_epsilon = epsilon;
                        best_classifier = classifier;
                    }
                }

                // Add to ensemble: F_t(x) = F_{t-1}(x) + alpha * h_t(x)
                double alpha = 0.5 * System.Math.Log(1 - min_epsilon / min_epsilon);
                mAlphaValues[t] = alpha;
                mWeakClassifiers[t] = best_classifier;

                // Update weights
                sample_index = 0;
                double weight_sum = 0;
                foreach (T rec in data_store)
                {
                    h = best_classifier.WeakPredict(rec);
                    y = (mClassVariableValue_YES == rec.Label ? 1 : -1);

                    weights [sample_index] = weights[sample_index] * System.Math.Exp(-h * alpha * y);
                    weight_sum = weights[sample_index];

                    sample_index++;
                }

                for (int i = 0; i < record_count; ++i)
                {
                    weights[sample_index] /= weight_sum;
                }
            }
        }

        public override string Predict(T rec)
        {
            double F_T = 0;
            for (int t = 0; t < mLayer; ++t)
            {
                WeakClassifier<T, U> classifier = mWeakClassifiers[t];
                double h = classifier.WeakPredict(rec);
                F_T += mAlphaValues[t] * h;
            }

            return F_T > 0 ? mClassVariableValue_YES : ClassVariableValue_No;
        }
    }
}
