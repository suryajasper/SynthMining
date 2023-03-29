import React from 'react';
import { fetchRequest } from '../../utils/utils';
import Popup, { PopupAttrs } from './popup';
import { TagAttrs } from '../project-loader';
import { withPopupStore } from '../hooks/popup-state';

interface EditTagDataSchema {
  tag: TagAttrs;
  projectId: string;
  uid: string;

  updateTag: (tagId: string, update: {
    name: string,
    description: string,
    goalQty: number,
  }) => void;
  removeTag: (tagId: string) => void;
}

export interface EditTagPopupAttrs extends PopupAttrs {
  active: boolean;
  data: EditTagDataSchema | undefined;
}

class BaseEditTagPopup extends Popup {
  private tagId: string | undefined;
  private uid: string | undefined;
  private projectId: string | undefined;

  private data: EditTagDataSchema;

  constructor(props: EditTagPopupAttrs) {
    super(props);
  }

  componentDidMount(): void {
    this.setState({
      actions: [
        { name: 'Save', res: this.saveTag.bind(this) },
        { name: 'Remove', res: this.removeTag.bind(this) },
      ],
    });
  }

  componentDidUpdate(): void {
    this.data = this.props.store.data as EditTagDataSchema;
    this.tagId = this.data?.tag?._id;

    this.projectId = this.props.data?.projectId;
    this.uid = this.props.data?.uid;
  }

  saveTag(): void {
    this.data.updateTag(this.tagId, {
      name: this.state.input['tagName'],
      description: this.state.input['tagDescription'],
      goalQty: parseInt(this.state.input['goalQty']),
    });
    this.props.store.hidePopup();
  }

  removeTag(): void {
    this.data.removeTag(this.tagId);
    this.props.store.hidePopup();
  }

  loadPopupContent(): JSX.Element[] {
    let tag = this.data?.tag;

    // this.setState({ title: `Edit ${tag?.name}` });

    return [
      this.createInputGroup({
        id: 'tagName',
        displayTitle: 'Name',
        initialValue: tag?.name || '',
      }),
      this.createInputGroup({
        id: 'goalQty',
        displayTitle: 'Goal Quantity',
        type: 'number',
        initialValue: tag?.goalQty?.toString() || '',
      }),
      this.createInputGroup({
        id: 'tagDescription',
        displayTitle: 'Description',
        type: 'textarea',
        initialValue: tag?.description || '',
      }),
    ];
  }
}

const EditTagPopup = withPopupStore(BaseEditTagPopup);
export { EditTagPopup };