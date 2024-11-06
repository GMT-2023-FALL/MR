import {Card, Col, List, Row, Statistic} from "antd";
import ModelViewerInListItem from "./ModelViewerInListItem.jsx";


const StatisticItem = (props) => {
    const {file_path, distance} = props;
    console.log(file_path, distance)
    return  <Col gutter={16}>
        <Row style={{ marginBottom: '8px' }}> {/* 设置底部间距 */}
            <Col span={24}>
                <Card bordered={false}>
                    <Statistic
                        title="Distance"
                        value={distance}
                        precision={2}
                        valueStyle={{
                            color: '#3f8600',
                            fontSize: '14px',
                        }}
                    />
                </Card>
            </Col>
        </Row>
        <Row> {/* 上下排列第二个统计项 */}
            <Col span={24}>
                <Card bordered={false}>
                    <Statistic
                        title="Path"
                        value={file_path}
                        precision={2}
                        valueStyle={{
                            color: '#cf1322',
                            fontSize: '10px',
                        }}
                        // prefix={<ArrowDownOutlined />}
                        // suffix="%"
                    />
                </Card>
            </Col>
        </Row>
    </Col>
}

const CustomListItem = (props) => {
    const {file_path, distance} = props;
    let genre = file_path.split('/')[0];
    let file_name = file_path.split('/')[1];
    let nw_path = '/normalized_database/' + genre + '/' + file_name + '.obj';

    return <List.Item style={{display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
        <Row style={{ width: '85%',display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <Col style={{ width: '40%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>  {/* 4比6的4部分 */}
                <StatisticItem {...props} />
            </Col>
            <Col style={{ width: '60%', height: '100%' }}>  {/* 4比6的6部分 */}
                <ModelViewerInListItem objFilePath={nw_path} />
            </Col>
        </Row>
    </List.Item>
}

const QueryResultPage = (props) => {
    const _data = props.data;
    return <List
        style={{padding: '24px',
            maxHeight: '100vh', // 设置最大高度
            width: '100%', // 设置宽度
            overflowY: 'auto', // 启用垂直滚动条   // 设置内边距
            }}
        itemLayout="vertical"
        size="large"
        pagination={false}
        dataSource={_data}
        footer={
            <div>
                <b>过肺回笼</b> 已经抽到底啦！
            </div>
        }
        renderItem={(item) => (
            <CustomListItem {...item}/>
        )}
    />
}

export default QueryResultPage;