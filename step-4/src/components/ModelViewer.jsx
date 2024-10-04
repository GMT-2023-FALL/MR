import React, {useEffect, useRef} from 'react';
import {Canvas} from '@react-three/fiber';
import {OBJLoader} from 'three/examples/jsm/loaders/OBJLoader';
import {OrbitControls} from '@react-three/drei';

const ModelViewer = ({ objFile }) => {
    const modelRef = useRef();

    useEffect(() => {
        if (objFile) {
            const loader = new OBJLoader();
            const reader = new FileReader();
            reader.onload = (event) => {
                const object = loader.parse(event.target.result);
                modelRef.current.clear(); // 清空之前的模型
                modelRef.current.add(object);
            };
            reader.readAsText(objFile);
        } else {
            if (modelRef.current){
                modelRef.current.clear(); // 如果 objFile 为 null，则清空模型
            }
        }
    }, [objFile]);

    return (
        <Canvas style={{width: '100%', height: '100%',backgroundColor: '#e0e0e0'}}>
            {/* 添加环境光以提供基础照明 */}
            <ambientLight intensity={0.3} />

            {/* 添加多个点光源以提高细节 */}
            <pointLight position={[5, 5, 5]} intensity={1} distance={50} />
            <pointLight position={[-5, -5, 5]} intensity={1} distance={50} />
            <pointLight position={[0, 5, -5]} intensity={1} distance={50} />
            <pointLight position={[0, -5, 5]} intensity={1} distance={50} />

            {/* 添加方向光源以模拟太阳光 */}
            <directionalLight position={[10, 10, 10]} intensity={0.5} />
            <directionalLight position={[-10, -10, -10]} intensity={0.5} />
            <group ref={modelRef}/>
            <OrbitControls/> {/* 添加 OrbitControls 组件 */}
        </Canvas>
    );
};

export default ModelViewer;