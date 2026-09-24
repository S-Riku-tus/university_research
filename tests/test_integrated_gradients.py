import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import sys
import tempfile
import json
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from utils.models.regression.base_regression import LogPowerCompression
from utils.explainability.spectrogram_explainers import (
    integrated_gradients,
    integrated_gradients_log_power,
)
from utils.explainability.training_integration import (
    _keras_attribution, _write_attribution_outputs, SUMMARY_HEADER,
    explain_keras_model,
    explainability_outputs_complete,
)


def log_model(weights):
    shape = (1, len(weights), 1)
    inputs = tf.keras.Input(shape=shape)
    z = LogPowerCompression()(inputs)
    out = tf.keras.layers.Dense(1, use_bias=False)(tf.keras.layers.Flatten()(z))
    model = tf.keras.Model(inputs, out)
    model.layers[-1].set_weights([np.array(weights, dtype=np.float32)[:, None]])
    return model


class IntegratedGradientsTest(unittest.TestCase):
    def test_log_dynamic_range_matches_each_analytic_signed_contribution(self):
        x=np.array([1e-12,1e-9,1e-6,1e-4],np.float32).reshape(1,4,1)
        weights=np.array([1,-2,3,-.5])
        model=log_model(weights)
        result,d=integrated_gradients(model,x,return_diagnostics=True)
        expected=weights*np.log1p(x.ravel().astype(float)/1e-12)
        np.testing.assert_allclose(result.ravel(),expected,rtol=2e-5,atol=1e-5)
        self.assertTrue(d['converged'])
        # The old 64-step endpoint trapezoid overestimates by orders of magnitude.
        a=tf.linspace(0.,1.,65)
        values=a[:,None,None,None]*x[None]
        with tf.GradientTape() as tape:
            tape.watch(values)
            y=model(values,training=False)
        grads=tape.gradient(y,values).numpy()
        old=x*np.mean((grads[:-1]+grads[1:])/2,axis=0)
        self.assertGreater(np.sum(abs(old.ravel()-expected))/np.sum(abs(expected)),100)

    def test_mixed_increasing_decreasing_nonzero_baseline(self):
        model=log_model([1,-2,3])
        x=np.array([1e-12,1e-4,1e-8],np.float32).reshape(1,3,1)
        b=np.array([1e-4,1e-12,1e-8],np.float32).reshape(1,3,1)
        result,d=integrated_gradients(model,x,b,return_diagnostics=True)
        expected=np.array([1,-2,3])*(np.log1p(x.ravel().astype(float)/1e-12)-np.log1p(b.ravel().astype(float)/1e-12))
        np.testing.assert_allclose(result.ravel(),expected,rtol=3e-5,atol=1e-5)
        self.assertTrue(d['converged'])

    def test_log_power_path_is_complete_and_records_distinct_semantics(self):
        inputs=tf.keras.Input((1,2,1))
        z=LogPowerCompression()(inputs)
        model=tf.keras.Model(inputs,(z[:,0,0,0]*z[:,0,1,0])[:,None])
        x=np.array([1e-6,1e-9],np.float32).reshape(1,2,1)
        a,d=integrated_gradients_log_power(model,x,steps=8,max_steps=64,
                                           return_diagnostics=True)
        z_value=np.log1p(x.ravel().astype(float)/1e-12)
        np.testing.assert_allclose(a.ravel(),np.prod(z_value)/2,rtol=2e-5)
        self.assertTrue(d['converged'])
        self.assertEqual(d['path_space'],'log_power')
        self.assertEqual(
            d['algorithm'],'log_power_straight_line_ig_gauss_legendre_v1')

    def test_linear_channels_and_batch_invariance(self):
        inputs=tf.keras.Input((2,2,2))
        model=tf.keras.Model(inputs,tf.reduce_sum(inputs*3,axis=(1,2,3))[:,None])
        x=np.arange(8,dtype=np.float32).reshape(2,2,2)
        b=np.ones_like(x)
        a=integrated_gradients(model,x,b,steps=8,max_steps=32,batch_size=1)
        c=integrated_gradients(model,x,b,steps=8,max_steps=32,batch_size=7)
        np.testing.assert_allclose(a,3*(x-b).sum(axis=-1),atol=1e-8)
        np.testing.assert_allclose(a,c,atol=1e-8)

    def test_nonlinear_interaction_preserves_raw_straight_line_path(self):
        inp=tf.keras.Input((1,2,1))
        z=LogPowerCompression()(inp)
        model=tf.keras.Model(inp,(z[:,0,0,0]*z[:,0,1,0])[:,None])
        x=np.array([1e-6,1e-9],np.float32).reshape(1,2,1)
        a,d=integrated_gradients(model,x,return_diagnostics=True)
        # Independent high-accuracy scalar quadrature in log(alpha), same path.
        from scipy.integrate import quad
        ratios=x.ravel().astype(float)/1e-12
        expected=[]
        for i in [0,1]:
            value=quad(lambda t: ratios[i]*np.exp(t)/(1+ratios[i]*np.exp(t))*
                       np.log1p(ratios[1-i]*np.exp(t)),-80,0,epsabs=1e-9)[0]
            expected.append(value)
        np.testing.assert_allclose(a.ravel(),expected,rtol=2e-5,atol=1e-5)
        self.assertTrue(d['converged'])

    def test_zero_delta_and_signed_cancellation(self):
        model=log_model([1,-1])
        x=np.full((1,2,1),1e-5,np.float32)
        a,d=integrated_gradients(model,x,return_diagnostics=True)
        self.assertTrue(d['converged'])
        self.assertGreater(a[0,0],10)
        self.assertAlmostEqual(a.sum(),0,places=6)
        zero,d=integrated_gradients(model,x,x,return_diagnostics=True)
        np.testing.assert_array_equal(zero,np.zeros((1,2)))
        self.assertTrue(d['converged'])

    def test_unconverged_is_reported_without_forcing_sum(self):
        model=log_model([1,2])
        x=np.array([1e-9,1e-4],np.float32).reshape(1,2,1)
        with self.assertWarns(RuntimeWarning):
            _,d=integrated_gradients(model,x,steps=2,max_steps=4,
                                    rtol=0,atol=0,map_rtol=0,return_diagnostics=True)
        self.assertFalse(d['converged'])
        self.assertNotEqual(d['completeness_error_model_units'],0)

    def test_heatflux_scaling_and_saved_diagnostics(self):
        model=log_model([1,-2])
        x=np.array([1e-7,1e-5],np.float32).reshape(1,2,1)
        scaler=MinMaxScaler().fit(np.array([[100000.],[900000.]]))
        config={'save_maps':False,'ig_steps':32,'ig_max_steps':256}
        values,signed,units,d=_keras_attribution('integrated_gradients',model,x,scaler,config,True)
        def predict(z):
            return scaler.inverse_transform(model(z,training=False).numpy()).ravel()
        yp=float(predict(x[None])[0])
        with tempfile.TemporaryDirectory() as directory:
            row=_write_attribution_outputs('integrated_gradients','test','sample',0,1,yp,values,predict,x,directory,3000,config,
                signed=signed,units=units,compute_curves=False,numerical_diagnostics=d)
            self.assertEqual(len(row),len(SUMMARY_HEADER))
            result=dict(zip(SUMMARY_HEADER,row))
            self.assertLess(result['completeness_relative_error'],1e-4)
            saved=json.loads((Path(directory)/'integrated_gradients_diagnostics.json').read_text())
            self.assertTrue(saved['converged'])

    def test_pipeline_records_primary_and_auxiliary_convergence(self):
        import pandas as pd
        model=log_model([1,-2])
        x=np.array([[[[1e-7],[1e-5]]]],np.float32)
        scaler=MinMaxScaler().fit(np.array([[100000.],[900000.]]))
        pred=scaler.inverse_transform(model(x).numpy()).ravel()
        before=[w.copy() for w in model.get_weights()]
        config={'enabled':True,'save_maps':False,'ig_steps':16,'ig_max_steps':256,
                'methods_by_model':{'test':['integrated_gradients']},
                'max_samples_per_fold':1,'curve_fractions':[0,1],
                'stability':{'enabled':True,'repeats':1},
                'sanity_check':{'enabled':True}}
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'explainability/fold1/test'
            output.mkdir(parents=True)
            explain_keras_model('test',model,scaler,x,pred,pred,0,str(output),3000,config)
            summary=pd.read_csv(output/'explainability_summary.csv')
            self.assertTrue(summary.ig_numerical_converged.all())
            stability=pd.read_csv(output/'input_stability.csv')
            self.assertTrue(stability.perturbed_ig_converged.all())
            sanity=pd.read_csv(output/'top_layer_randomization_sanity.csv')
            self.assertTrue(sanity.randomized_ig_converged.all())
            self.assertTrue(explainability_outputs_complete(directory,config,['test'],1))
            summary.drop(columns=['ig_numerical_converged','ig_nodes']).to_csv(output/'explainability_summary.csv',index=False)
            self.assertFalse(explainability_outputs_complete(directory,config,['test'],1))
        for old,new in zip(before,model.get_weights()):
            np.testing.assert_array_equal(old,new)


if __name__=='__main__':
    unittest.main()
