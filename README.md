# PDEVis #

This is visualization system for supporting the process of deriving Partial Differential Equation(PDE) from sampling dataset.

### How do I get set up? ###

The python virtual environment should be built for repos.
```
(MacOS) python3 -m venv env
```


### Environment ###
```
. env/bin/activate
pip install -r requirements.txt
```

### Backend ###
```
flask run
```

### Frontend ###
```
npm install -g @vue/cli@3.7.0
vue create client
npm install bootstrap-vue
npm i -S bootstrap
npm install vue-awesome
npm install axios
npm install echarts@4.9.0
npm install echarts-gl@1.1.2
npm run serve
```