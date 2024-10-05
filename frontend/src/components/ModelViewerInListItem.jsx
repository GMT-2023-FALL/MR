import React, {useEffect} from 'react';
import {Canvas} from '@react-three/fiber';
import {OBJLoader} from 'three/examples/jsm/loaders/OBJLoader';
import {OrbitControls} from '@react-three/drei';

const ModelViewerInListItem = ({objFilePath}) => {
    const [model, setModel] = React.useState(null);
    const modelRef = React.useRef();

    useEffect(() => {
        if (objFilePath) {
            const loader = new OBJLoader();
            loader.load(objFilePath,
                (object) => {
                    console.log(objFilePath, object, typeof object);
                    setModel(object);
                    if (modelRef.current) {
                        modelRef.current.clear();
                        modelRef.current.add(object);
                    }
                },
                (xhr) => {
                    console.log((xhr.loaded / xhr.total * 100) + '% loaded');
                },
                (error) => {
                    console.log('An error happened: ' + error);
                });
        }
    }, [objFilePath]);

    return (
        <Canvas style={{width: '100%', height: '100%', backgroundColor: '#e0e0e0'}}>
            {/* 添加环境光以提供基础照明 */}
            <ambientLight intensity={0.3}/>

            {/* 添加多个点光源以提高细节 */}
            <pointLight position={[5, 5, 5]} intensity={1} distance={50}/>
            <pointLight position={[-5, -5, 5]} intensity={1} distance={50}/>
            <pointLight position={[0, 5, -5]} intensity={1} distance={50}/>
            <pointLight position={[0, -5, 5]} intensity={1} distance={50}/>

            {/* 添加方向光源以模拟太阳光 */}
            <directionalLight position={[10, 10, 10]} intensity={0.5}/>
            <directionalLight position={[-10, -10, -10]} intensity={0.5}/>
            {/*<group ref={modelRef}/>*/}
            {model && <primitive object={model} ref={modelRef}/>}
            <OrbitControls/> {/* 添加 OrbitControls 组件 */}
        </Canvas>
    );
};

export default ModelViewerInListItem;