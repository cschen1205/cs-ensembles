using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SimuKit.ML.Solvers;
using SimuKit.ML.Lang;

namespace SimuKit.ML.Ensemble
{
    public class EnsembleLearning<T, U> : Classifier<T, U>
        where T : DataRecord<U>
    {
        protected WeakClassifier<T, U>[] mWeakClassifiers = null;
        public WeakClassifier<T, U>[] WeakClassifiers
        {
            get { return mWeakClassifiers; }
        }


    }
}
