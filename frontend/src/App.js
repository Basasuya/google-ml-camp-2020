import React from "react";
// import * as d3 from "d3"
import "./App.css";
import { Col, Row } from "antd";
import Snapshots from "./snapshots";
import ImageList from "./imagelist";
import ImageDecider from "./imagedecider";
import ImgModal from "./img-modal";
// import Graph from "./graph"
import axios from 'axios';

class App extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            snapshotHeight: 0,
            enableModal: false,
            snapshots: [],
            imagelist: [],
            imageChoose: { "path": "http://0.0.0.0:8000/file/choose.png", "index": -1}
        }
    }

    handleLasso(chooseList) {
        // console.log(chooseList)
        let result = []
        for(let i = 0; i < chooseList.length; ++i) {
            result.push({"path" : "http://0.0.0.0:8000/" + this.state.snapshots[chooseList[i]]["path"],"index": this.state.snapshots[chooseList[i]]["key"]})
        }
        console.log(result)
        this.setState({ "imagelist" : result})

    }

    handleSetHeight(height) {
      this.setState({snapshotHeight: height});
    }

    changeSearch(id) {
        let tmp = this.state.imagelist[id]["index"];
        const imgUrl = `http://0.0.0.0:8000/${this.state.snapshots[tmp]["path"]}`;
        this.setState({ "imageChoose" : {
            "path" : imgUrl,
            "index": this.state.snapshots[tmp]["key"],
          }
        });
        this.handleOpenModal();
        // console.log("hhhh", id)
    }

    handleKnn(id, k) {
        // console.log("knn", id, k)
        if(id == -1) return

        axios
        .post('http://localhost:5000/getKNN', {'id': id, 'k': k}, { Accept: 'application/json', 'Content-Type': 'application/json'})
        .then((response) => {
            console.log('hhhh')
            console.log(response.data)
            this.handleLasso(response.data.result)
            // this.setState({ graph: response.data.result})
        })
    }

    componentDidMount() {
        axios
        .get('http://localhost:5000/getTSNE', { Accept: 'application/json', 'Content-Type': 'application/json'})
        .then((response) => {
            this.setState({ snapshots: response.data})
            
        })

        // axios
        // .get('http://localhost:5000/getTSNE', { Accept: 'application/json', 'Content-Type': 'application/json','Access-Control-Allow-Origin': '*'})
        // .then((response) => {
        //     this.setState({ snapshots: response.data})
        // })

        // d3.json("./test_data.json").then(snapshots => {
        //     this.setState({
        //         snapshots: snapshots,
        //         graph: snapshots[0].graph
        //     })
        // })
    }

    handleCloseModal() {
      this.setState({
        enableModal: false
      })
    }

    handleOpenModal() {
      this.setState({
        enableModal: true
      })
    }

    render() {
        // const snapshots = this.state.snapshots
        // console.log('render!');
        console.log("kkkk", this.state)
        // const { graph } = this.state
        return (
            <div className="App">
                <Row>
                    <Col span={15}>
                        <Snapshots snapshots={this.state.snapshots} onLasso={this.handleLasso.bind(this)}
                          onSetHeight={this.handleSetHeight.bind(this)}
                        />
                    </Col>
                    <Col span={9}>
                        <Row span={9}>
                         <ImageDecider imageChoose={this.state.imageChoose} onKnn={this.handleKnn.bind(this)}/>
                        </Row>
                        <Row span={15}>
                            <ImageList onChange={this.changeSearch.bind(this)} imagelist={this.state.imagelist} totalHeight={this.state.snapshotHeight}/>
                        </Row>
                        {/* <Graph graph={graph} onHighLight={this.handleHighLight.bind(this)}/> */}
                    </Col>
                </Row>
                <ImgModal
                  enableModal={this.state.enableModal}
                  src={this.state.imageChoose.path}
                  onCloseModal={this.handleCloseModal.bind(this)}
                  onOpenModal={this.handleOpenModal.bind(this)}
                />
            </div>
        )
    }
}

export default App
