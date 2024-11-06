// src/App.js
import React, {useState} from 'react';
import {Empty, Flex, Layout, message, Row, Select, Slider, Space, Spin, Splitter, Typography} from 'antd';
import {InboxOutlined} from '@ant-design/icons';
import Dragger from "antd/lib/upload/Dragger.js";
import ModelViewer from "../components/ModelViewer.jsx";
import QueryResultPage from "../components/QueryResultPage.jsx";

// const API_URL = 'http://localhost:8000/query';
const API_URL = 'api/query';


const {Header, Content, Footer} = Layout;

function QueryingPage() {
    const [file, setFile] = useState(null);
    const [count, setCount] = useState(5);
    const [distanceThreshold, setDistanceThreshold] = useState(0.1);
    const [resultQuery, setResultQuery] = useState(null);
    const [spinning, setSpinning] = useState(false);
    const [method, setMethod] = useState('KNN');

    const handleOnChangeComplete = async (number) => {
        if(!file){
            setCount(number);
            return
        }
        setSpinning(true);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('count', number.toString()); // 添加 count 参数
        formData.append('method', method); // 添加 t 参数
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const result = await response.json();
                console.log('result:', result, typeof result);
                setFile(file);
                setResultQuery(result);
                setCount(number);
                // onSuccess(result); // 成功回调
                message.success(`${file.name} file uploaded successfully`);
                setSpinning(false);
            } else {
                throw new Error('上传失败');
            }
        } catch (error) {
            setSpinning(false);
            console.log('上传失败:', error);
            // onError(error); // 失败回调
            message.error(`${file.name} file upload failed: ${error.message}`);
        }
    }

    const handleMethodOnChange = async (nwMethod) => {
        console.log('method:', nwMethod);
        if (!file) {
            setMethod(nwMethod);
            return
        }
        setSpinning(true);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('count', count.toString()); // 添加 count 参数
        formData.append('method', nwMethod); // 添加 t 参数
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const result = await response.json();
                console.log('result:', result, typeof result);
                setFile(file);
                setResultQuery(result);
                setMethod(nwMethod);
                // onSuccess(result); // 成功回调
                message.success(`${file.name} file uploaded successfully`);
                setSpinning(false);
            } else {
                throw new Error('上传失败');
            }
        } catch (error) {
            setSpinning(false);
            console.log('上传失败:', error);
            // onError(error); // 失败回调
            message.error(`${file.name} file upload failed: ${error.message}`);
        }
    }

    const customRequest = async ({file, onSuccess, onError}) => {
        setSpinning(true);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('count', count.toString()); // 添加 count 参数
        formData.append('method', method); // 添加 t 参数
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const result = await response.json();
                setTimeout(() => {
                    setFile(file);
                    setResultQuery(result);
                    onSuccess(result); // 成功回调
                    setSpinning(false);
                    message.success(`${file.name} file uploaded successfully`);
                }, 3000);
            } else {
                // onError('上传失败');
                throw new Error('上传失败');
            }
        } catch (error) {
            console.log('上传失败:', error);
            setSpinning(false);
            onError(error); // 失败回调
            message.error(`${file.name} file upload failed: ${error.message}`);
        }
    };

    const handleRemove = (file) => {
        console.log('file removed:', file);
        setFile(null);
        setResultQuery(null);
    };

    const uploaderProps = {
        name: 'file',
        accept: '.obj',
        multiple: false,
        customRequest: customRequest,
        showUploadList: {
            showPreviewIcon: false,
            showRemoveIcon: true,
            showDownloadIcon: false,
        },
        onRemove: handleRemove,
        // onChange: handleUpload,
        fileList: file ? [file] : [],
        onChange(info) {
            console.log('onChange:', info);
            if (info.file.status !== 'uploading') {
                console.log(info.file, info.fileList);
            }
            if (info.file.status === 'done') {
                message.success(`${info.file.name} file uploaded successfully`);
            } else if (info.file.status === 'error') {
                message.error(`${info.file.name} file upload failed.`);
            }
        },
        onDrop(e) {
            console.log('Dropped files', e.dataTransfer.files);
        },
    };

    return (
        <Layout style={{minHeight: '100vh', width: 'auto'}}>
            <Spin percent={'auto'} size="large" spinning={spinning}>
            <Header style={{background: 'transparent', marginBottom: '25px'}}>
                <h1 style={{
                    backgroundColor:'transparent',

                }}>MR - Querying Application</h1>
            </Header>
            <Content style={{padding: '20px', minHeight: '100vh'}}>
                <Splitter
                    style={{
                        minHeight: '100vh',
                        minWidth: '100vh',
                        boxShadow: '0 0 10px rgba(0, 0, 0, 0.1)',
                    }}
                >
                    <Splitter.Panel>
                        <Splitter layout="vertical">
                            <Splitter.Panel collapsible={false} defaultSize={'40%'}>
                                <div style={{
                                    display: 'flex',
                                    flexDirection: 'column',
                                    gap: '10px',
                                    height: 'auto',
                                    justifyContent: 'center',
                                    alignItems: 'center',
                                    padding: '20px',

                                }}>
                                    {/*h5 and select should be in a row*/}
                                    <h5>K best-matching shapes</h5>
                                    <Select
                                        defaultValue="default"
                                        value={method}
                                        style={{width: 120}}
                                        onChange={handleMethodOnChange}
                                        options={[
                                            {value: 'default', label: 'default'},
                                            {value: 'KNN', label: 'KNN'},
                                            {value: 'KDTree', label: 'KDTree'},
                                        ]}
                                    />
                                    <Slider
                                        style={{width: '80%'}}
                                        step={1}
                                        min={1}
                                        max={10}
                                        marks={
                                            {
                                                1: '1',
                                                5: '5',
                                                10: '10',
                                            }
                                        }
                                        defaultValue={count}
                                        onChangeComplete={handleOnChangeComplete}/>
                                    <Dragger {...uploaderProps}>
                                        <p className="ant-upload-drag-icon">
                                            <InboxOutlined/>
                                        </p>
                                        <p className="ant-upload-text">Click or drag file to this area to upload</p>
                                        <p className="ant-upload-hint">
                                            Support for a single or bulk upload. Strictly prohibited from uploading
                                            company
                                            data
                                            or
                                            other
                                            banned files.
                                        </p>
                                    </Dragger>
                                </div>
                            </Splitter.Panel>
                            <Splitter.Panel collapsible={false} defaultSize={'60%'}>
                                {/*<Desc text="Bottom"/>*/}
                                <ModelViewer objFile={file}/>
                            </Splitter.Panel>
                        </Splitter>
                    </Splitter.Panel>
                    <Splitter.Panel collapsible={false}>
                        {/*<Desc text="right"/>*/}
                        {resultQuery ? <div>
                            <QueryResultPage data={resultQuery}/>

                        </div> : <div style={{
                            display: 'flex',
                            gap: '10px',
                            height: 'auto',
                            justifyContent: 'center',
                            alignItems: 'center',
                            padding: '20px',

                        }}>
                            <Empty style={{
                                height: '100vh',
                                display: 'flex',
                                flexDirection: 'row',
                                justifyContent: 'center',
                                alignItems: 'center',
                            }}/>
                        </div>}
                    </Splitter.Panel>
                </Splitter>

            </Content>
            <Footer style={{textAlign: 'center'}}>
                ©2024 扎不多得嘞 - Querying Application - 噪称冯得
            </Footer>
            </Spin>
        </Layout>
    );
}

export default QueryingPage;
