using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Solvers;
using Lang;

namespace Ensemble
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
