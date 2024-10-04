// src/App.js
import React, {useState} from 'react';
import {Empty, Flex, Layout, message, Slider, Spin, Splitter, Typography} from 'antd';
import {InboxOutlined} from '@ant-design/icons';
import Dragger from "antd/lib/upload/Dragger.js";
import ModelViewer from "../components/ModelViewer.jsx";
import QueryResultPage from "../components/QueryResultPage.jsx";


const {Header, Content, Footer} = Layout;

function QueryingPage() {
    const [file, setFile] = useState(null);
    const [count, setCount] = useState(3);
    const [distanceThreshold, setDistanceThreshold] = useState(0.1);
    const [resultQuery, setResultQuery] = useState(null);
    const [spinning, setSpinning] = useState(false);

    const customRequest = async ({file, onSuccess, onError}) => {
        console.log('customRequest:', file, typeof file);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('count', count.toString()); // 添加 count 参数
        formData.append('t', distanceThreshold.toString()); // 添加 t 参数
        try {
            const response = await fetch('http://localhost:8000/query', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const result = await response.json();
                console.log('result:', result, typeof result);
                setFile(file);
                setResultQuery(result);
                onSuccess(result); // 成功回调
                // message.success(`${file.name} 文件上传成功`);
                message.success(`${file.name} file uploaded successfully`);
            } else {
                // onError('上传失败');
                throw new Error('上传失败');
            }
        } catch (error) {
            console.log('上传失败:', error);
            onError(error); // 失败回调
            // message.error(`${file.name} 文件上传失败: ${error.message}`);
            message.error(`${file.name} file upload failed: ${error.message}`);
        }
    };

    const handleUpload = (info) => {
        if (info.file.status !== 'uploading') {
            console.log(info.file, info.fileList);
        }
        if (info.file.status === 'done') {
            message.success(`${info.file.name} file uploaded successfully`);
        } else if (info.file.status === 'error') {
            message.error(`${info.file.name} file upload failed.`);
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

    const Desc = (props) => (
        <Flex
            justify="center"
            align="center"
            style={{
                height: '100%',
            }}
        >
            <Typography.Title
                type="secondary"
                level={5}
                style={{
                    whiteSpace: 'nowrap',
                }}
            >
                {props.text}
            </Typography.Title>
        </Flex>
    );

    return (
        <Layout style={{minHeight: '100vh', width: 'auto'}}>
            <Spin percent={'auto'} size="large" spinning={spinning}>
            <Header style={{background: 'transparent', marginBottom: '25px'}}>
                <h1 style={{
                    backgroundColor:'transparent',

                }}>以图搜图 - .obj 文件上传与搜索</h1>
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
                                    <h5>K best-matching shapes</h5>
                                    <Slider
                                        style={{width: '80%'}}
                                        step={1}
                                        min={1}
                                        max={10}
                                        marks={
                                            {
                                                1: '1',
                                                3: '3',
                                                10: '10',
                                            }
                                        }
                                        defaultValue={3}
                                        onChangeComplete={(number) => {
                                            console.log('count:', number);
                                            setCount(number);
                                            setSpinning(true);
                                            setTimeout(() => {
                                                setSpinning(false);
                                            }, 3000);
                                        }}/>
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
                React Ant Design 以图搜图界面 ©2024
            </Footer>
            </Spin>
        </Layout>
    );
}

export default QueryingPage;
