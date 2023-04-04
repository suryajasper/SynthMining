import React from 'react';
import { fetchRequest } from '../../utils/utils';
import Popup, { PopupAttrs } from './popup';
import { TagAttrs } from '../project-loader';
import { PopupManagerState, withPopupStore } from '../hooks/popup-state';

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

class BaseEditTagPopup extends Popup<EditTagDataSchema> {
  private tagId: string | undefined;
  private uid: string | undefined;
  private projectId: string | undefined;

  private data: EditTagDataSchema;

  constructor(props: PopupAttrs<EditTagDataSchema>) {
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

  override onPopupActivate(): void {
    this.data = this.props.store.data as EditTagDataSchema;
    console.log('update', this.data);
    this.tagId = this.data?.tag?._id;

    this.projectId = this.props.data?.projectId;
    this.uid = this.props.data?.uid;

    this.setState({ title: `Edit Tag "${this.data.tag?.name}"` });
  }

  override loadPopupContent(): JSX.Element[] {
    let tag = this.data?.tag;
    console.log('loading popup content', tag);

    return [
      this.createInputGroup({
        id: 'tagName',
        placeholder: 'Tag Name',
        displayTitle: 'Name',
        initialValue: tag?.name || '',
      }),
      this.createInputGroup({
        id: 'goalQty',
        placeholder: '200',
        displayTitle: 'Goal Quantity',
        type: 'number',
        initialValue: tag?.goalQty?.toString() || '',
      }),
      this.createInputGroup({
        id: 'tagDescription',
        placeholder: 'Brief description of your tag for other users',
        displayTitle: 'Description',
        type: 'textarea',
        initialValue: tag?.description || '',
      }),
    ];
  }
}

const EditTagPopup = withPopupStore(BaseEditTagPopup);
export { EditTagDataSchema, EditTagPopup };