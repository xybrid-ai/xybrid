// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'decision.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
    'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models');

/// @nodoc
mixin _$FfiDecisionContent {
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String value) text,
    required TResult Function(String jsonText) json,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String value)? text,
    TResult? Function(String jsonText)? json,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String value)? text,
    TResult Function(String jsonText)? json,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionContent_Text value) text,
    required TResult Function(FfiDecisionContent_Json value) json,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionContent_Text value)? text,
    TResult? Function(FfiDecisionContent_Json value)? json,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionContent_Text value)? text,
    TResult Function(FfiDecisionContent_Json value)? json,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $FfiDecisionContentCopyWith<$Res> {
  factory $FfiDecisionContentCopyWith(
          FfiDecisionContent value, $Res Function(FfiDecisionContent) then) =
      _$FfiDecisionContentCopyWithImpl<$Res, FfiDecisionContent>;
}

/// @nodoc
class _$FfiDecisionContentCopyWithImpl<$Res, $Val extends FfiDecisionContent>
    implements $FfiDecisionContentCopyWith<$Res> {
  _$FfiDecisionContentCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of FfiDecisionContent
  /// with the given fields replaced by the non-null parameter values.
}

/// @nodoc
abstract class _$$FfiDecisionContent_TextImplCopyWith<$Res> {
  factory _$$FfiDecisionContent_TextImplCopyWith(
          _$FfiDecisionContent_TextImpl value,
          $Res Function(_$FfiDecisionContent_TextImpl) then) =
      __$$FfiDecisionContent_TextImplCopyWithImpl<$Res>;
  @useResult
  $Res call({String value});
}

/// @nodoc
class __$$FfiDecisionContent_TextImplCopyWithImpl<$Res>
    extends _$FfiDecisionContentCopyWithImpl<$Res,
        _$FfiDecisionContent_TextImpl>
    implements _$$FfiDecisionContent_TextImplCopyWith<$Res> {
  __$$FfiDecisionContent_TextImplCopyWithImpl(
      _$FfiDecisionContent_TextImpl _value,
      $Res Function(_$FfiDecisionContent_TextImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiDecisionContent
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? value = null,
  }) {
    return _then(_$FfiDecisionContent_TextImpl(
      value: null == value
          ? _value.value
          : value // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class _$FfiDecisionContent_TextImpl extends FfiDecisionContent_Text {
  const _$FfiDecisionContent_TextImpl({required this.value}) : super._();

  @override
  final String value;

  @override
  String toString() {
    return 'FfiDecisionContent.text(value: $value)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiDecisionContent_TextImpl &&
            (identical(other.value, value) || other.value == value));
  }

  @override
  int get hashCode => Object.hash(runtimeType, value);

  /// Create a copy of FfiDecisionContent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiDecisionContent_TextImplCopyWith<_$FfiDecisionContent_TextImpl>
      get copyWith => __$$FfiDecisionContent_TextImplCopyWithImpl<
          _$FfiDecisionContent_TextImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String value) text,
    required TResult Function(String jsonText) json,
  }) {
    return text(value);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String value)? text,
    TResult? Function(String jsonText)? json,
  }) {
    return text?.call(value);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String value)? text,
    TResult Function(String jsonText)? json,
    required TResult orElse(),
  }) {
    if (text != null) {
      return text(value);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionContent_Text value) text,
    required TResult Function(FfiDecisionContent_Json value) json,
  }) {
    return text(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionContent_Text value)? text,
    TResult? Function(FfiDecisionContent_Json value)? json,
  }) {
    return text?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionContent_Text value)? text,
    TResult Function(FfiDecisionContent_Json value)? json,
    required TResult orElse(),
  }) {
    if (text != null) {
      return text(this);
    }
    return orElse();
  }
}

abstract class FfiDecisionContent_Text extends FfiDecisionContent {
  const factory FfiDecisionContent_Text({required final String value}) =
      _$FfiDecisionContent_TextImpl;
  const FfiDecisionContent_Text._() : super._();

  String get value;

  /// Create a copy of FfiDecisionContent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiDecisionContent_TextImplCopyWith<_$FfiDecisionContent_TextImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiDecisionContent_JsonImplCopyWith<$Res> {
  factory _$$FfiDecisionContent_JsonImplCopyWith(
          _$FfiDecisionContent_JsonImpl value,
          $Res Function(_$FfiDecisionContent_JsonImpl) then) =
      __$$FfiDecisionContent_JsonImplCopyWithImpl<$Res>;
  @useResult
  $Res call({String jsonText});
}

/// @nodoc
class __$$FfiDecisionContent_JsonImplCopyWithImpl<$Res>
    extends _$FfiDecisionContentCopyWithImpl<$Res,
        _$FfiDecisionContent_JsonImpl>
    implements _$$FfiDecisionContent_JsonImplCopyWith<$Res> {
  __$$FfiDecisionContent_JsonImplCopyWithImpl(
      _$FfiDecisionContent_JsonImpl _value,
      $Res Function(_$FfiDecisionContent_JsonImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiDecisionContent
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? jsonText = null,
  }) {
    return _then(_$FfiDecisionContent_JsonImpl(
      jsonText: null == jsonText
          ? _value.jsonText
          : jsonText // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class _$FfiDecisionContent_JsonImpl extends FfiDecisionContent_Json {
  const _$FfiDecisionContent_JsonImpl({required this.jsonText}) : super._();

  @override
  final String jsonText;

  @override
  String toString() {
    return 'FfiDecisionContent.json(jsonText: $jsonText)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiDecisionContent_JsonImpl &&
            (identical(other.jsonText, jsonText) ||
                other.jsonText == jsonText));
  }

  @override
  int get hashCode => Object.hash(runtimeType, jsonText);

  /// Create a copy of FfiDecisionContent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiDecisionContent_JsonImplCopyWith<_$FfiDecisionContent_JsonImpl>
      get copyWith => __$$FfiDecisionContent_JsonImplCopyWithImpl<
          _$FfiDecisionContent_JsonImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String value) text,
    required TResult Function(String jsonText) json,
  }) {
    return json(jsonText);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String value)? text,
    TResult? Function(String jsonText)? json,
  }) {
    return json?.call(jsonText);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String value)? text,
    TResult Function(String jsonText)? json,
    required TResult orElse(),
  }) {
    if (json != null) {
      return json(jsonText);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionContent_Text value) text,
    required TResult Function(FfiDecisionContent_Json value) json,
  }) {
    return json(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionContent_Text value)? text,
    TResult? Function(FfiDecisionContent_Json value)? json,
  }) {
    return json?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionContent_Text value)? text,
    TResult Function(FfiDecisionContent_Json value)? json,
    required TResult orElse(),
  }) {
    if (json != null) {
      return json(this);
    }
    return orElse();
  }
}

abstract class FfiDecisionContent_Json extends FfiDecisionContent {
  const factory FfiDecisionContent_Json({required final String jsonText}) =
      _$FfiDecisionContent_JsonImpl;
  const FfiDecisionContent_Json._() : super._();

  String get jsonText;

  /// Create a copy of FfiDecisionContent
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiDecisionContent_JsonImplCopyWith<_$FfiDecisionContent_JsonImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
mixin _$FfiDecisionInstructions {
  FfiDecisionContent get value => throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(FfiDecisionContent value) content,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionContent value)? content,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(FfiDecisionContent value)? content,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionInstructions_Content value) content,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionInstructions_Content value)? content,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionInstructions_Content value)? content,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;

  /// Create a copy of FfiDecisionInstructions
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $FfiDecisionInstructionsCopyWith<FfiDecisionInstructions> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $FfiDecisionInstructionsCopyWith<$Res> {
  factory $FfiDecisionInstructionsCopyWith(FfiDecisionInstructions value,
          $Res Function(FfiDecisionInstructions) then) =
      _$FfiDecisionInstructionsCopyWithImpl<$Res, FfiDecisionInstructions>;
  @useResult
  $Res call({FfiDecisionContent value});

  $FfiDecisionContentCopyWith<$Res> get value;
}

/// @nodoc
class _$FfiDecisionInstructionsCopyWithImpl<$Res,
        $Val extends FfiDecisionInstructions>
    implements $FfiDecisionInstructionsCopyWith<$Res> {
  _$FfiDecisionInstructionsCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of FfiDecisionInstructions
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? value = null,
  }) {
    return _then(_value.copyWith(
      value: null == value
          ? _value.value
          : value // ignore: cast_nullable_to_non_nullable
              as FfiDecisionContent,
    ) as $Val);
  }

  /// Create a copy of FfiDecisionInstructions
  /// with the given fields replaced by the non-null parameter values.
  @override
  @pragma('vm:prefer-inline')
  $FfiDecisionContentCopyWith<$Res> get value {
    return $FfiDecisionContentCopyWith<$Res>(_value.value, (value) {
      return _then(_value.copyWith(value: value) as $Val);
    });
  }
}

/// @nodoc
abstract class _$$FfiDecisionInstructions_ContentImplCopyWith<$Res>
    implements $FfiDecisionInstructionsCopyWith<$Res> {
  factory _$$FfiDecisionInstructions_ContentImplCopyWith(
          _$FfiDecisionInstructions_ContentImpl value,
          $Res Function(_$FfiDecisionInstructions_ContentImpl) then) =
      __$$FfiDecisionInstructions_ContentImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({FfiDecisionContent value});

  @override
  $FfiDecisionContentCopyWith<$Res> get value;
}

/// @nodoc
class __$$FfiDecisionInstructions_ContentImplCopyWithImpl<$Res>
    extends _$FfiDecisionInstructionsCopyWithImpl<$Res,
        _$FfiDecisionInstructions_ContentImpl>
    implements _$$FfiDecisionInstructions_ContentImplCopyWith<$Res> {
  __$$FfiDecisionInstructions_ContentImplCopyWithImpl(
      _$FfiDecisionInstructions_ContentImpl _value,
      $Res Function(_$FfiDecisionInstructions_ContentImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiDecisionInstructions
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? value = null,
  }) {
    return _then(_$FfiDecisionInstructions_ContentImpl(
      value: null == value
          ? _value.value
          : value // ignore: cast_nullable_to_non_nullable
              as FfiDecisionContent,
    ));
  }
}

/// @nodoc

class _$FfiDecisionInstructions_ContentImpl
    extends FfiDecisionInstructions_Content {
  const _$FfiDecisionInstructions_ContentImpl({required this.value})
      : super._();

  @override
  final FfiDecisionContent value;

  @override
  String toString() {
    return 'FfiDecisionInstructions.content(value: $value)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiDecisionInstructions_ContentImpl &&
            (identical(other.value, value) || other.value == value));
  }

  @override
  int get hashCode => Object.hash(runtimeType, value);

  /// Create a copy of FfiDecisionInstructions
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiDecisionInstructions_ContentImplCopyWith<
          _$FfiDecisionInstructions_ContentImpl>
      get copyWith => __$$FfiDecisionInstructions_ContentImplCopyWithImpl<
          _$FfiDecisionInstructions_ContentImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(FfiDecisionContent value) content,
  }) {
    return content(value);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionContent value)? content,
  }) {
    return content?.call(value);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(FfiDecisionContent value)? content,
    required TResult orElse(),
  }) {
    if (content != null) {
      return content(value);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionInstructions_Content value) content,
  }) {
    return content(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionInstructions_Content value)? content,
  }) {
    return content?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionInstructions_Content value)? content,
    required TResult orElse(),
  }) {
    if (content != null) {
      return content(this);
    }
    return orElse();
  }
}

abstract class FfiDecisionInstructions_Content extends FfiDecisionInstructions {
  const factory FfiDecisionInstructions_Content(
          {required final FfiDecisionContent value}) =
      _$FfiDecisionInstructions_ContentImpl;
  const FfiDecisionInstructions_Content._() : super._();

  @override
  FfiDecisionContent get value;

  /// Create a copy of FfiDecisionInstructions
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiDecisionInstructions_ContentImplCopyWith<
          _$FfiDecisionInstructions_ContentImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
mixin _$FfiDecisionQuestionKind {
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(List<FfiDecisionOption> options) choice,
    required TResult Function(FfiDecisionContent? yes, FfiDecisionContent? no)
        yesNo,
    required TResult Function(List<FfiDecisionLevel> levels) score,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(List<FfiDecisionOption> options)? choice,
    TResult? Function(FfiDecisionContent? yes, FfiDecisionContent? no)? yesNo,
    TResult? Function(List<FfiDecisionLevel> levels)? score,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(List<FfiDecisionOption> options)? choice,
    TResult Function(FfiDecisionContent? yes, FfiDecisionContent? no)? yesNo,
    TResult Function(List<FfiDecisionLevel> levels)? score,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionQuestionKind_Choice value) choice,
    required TResult Function(FfiDecisionQuestionKind_YesNo value) yesNo,
    required TResult Function(FfiDecisionQuestionKind_Score value) score,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionQuestionKind_Choice value)? choice,
    TResult? Function(FfiDecisionQuestionKind_YesNo value)? yesNo,
    TResult? Function(FfiDecisionQuestionKind_Score value)? score,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionQuestionKind_Choice value)? choice,
    TResult Function(FfiDecisionQuestionKind_YesNo value)? yesNo,
    TResult Function(FfiDecisionQuestionKind_Score value)? score,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $FfiDecisionQuestionKindCopyWith<$Res> {
  factory $FfiDecisionQuestionKindCopyWith(FfiDecisionQuestionKind value,
          $Res Function(FfiDecisionQuestionKind) then) =
      _$FfiDecisionQuestionKindCopyWithImpl<$Res, FfiDecisionQuestionKind>;
}

/// @nodoc
class _$FfiDecisionQuestionKindCopyWithImpl<$Res,
        $Val extends FfiDecisionQuestionKind>
    implements $FfiDecisionQuestionKindCopyWith<$Res> {
  _$FfiDecisionQuestionKindCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of FfiDecisionQuestionKind
  /// with the given fields replaced by the non-null parameter values.
}

/// @nodoc
abstract class _$$FfiDecisionQuestionKind_ChoiceImplCopyWith<$Res> {
  factory _$$FfiDecisionQuestionKind_ChoiceImplCopyWith(
          _$FfiDecisionQuestionKind_ChoiceImpl value,
          $Res Function(_$FfiDecisionQuestionKind_ChoiceImpl) then) =
      __$$FfiDecisionQuestionKind_ChoiceImplCopyWithImpl<$Res>;
  @useResult
  $Res call({List<FfiDecisionOption> options});
}

/// @nodoc
class __$$FfiDecisionQuestionKind_ChoiceImplCopyWithImpl<$Res>
    extends _$FfiDecisionQuestionKindCopyWithImpl<$Res,
        _$FfiDecisionQuestionKind_ChoiceImpl>
    implements _$$FfiDecisionQuestionKind_ChoiceImplCopyWith<$Res> {
  __$$FfiDecisionQuestionKind_ChoiceImplCopyWithImpl(
      _$FfiDecisionQuestionKind_ChoiceImpl _value,
      $Res Function(_$FfiDecisionQuestionKind_ChoiceImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiDecisionQuestionKind
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? options = null,
  }) {
    return _then(_$FfiDecisionQuestionKind_ChoiceImpl(
      options: null == options
          ? _value._options
          : options // ignore: cast_nullable_to_non_nullable
              as List<FfiDecisionOption>,
    ));
  }
}

/// @nodoc

class _$FfiDecisionQuestionKind_ChoiceImpl
    extends FfiDecisionQuestionKind_Choice {
  const _$FfiDecisionQuestionKind_ChoiceImpl(
      {required final List<FfiDecisionOption> options})
      : _options = options,
        super._();

  final List<FfiDecisionOption> _options;
  @override
  List<FfiDecisionOption> get options {
    if (_options is EqualUnmodifiableListView) return _options;
    // ignore: implicit_dynamic_type
    return EqualUnmodifiableListView(_options);
  }

  @override
  String toString() {
    return 'FfiDecisionQuestionKind.choice(options: $options)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiDecisionQuestionKind_ChoiceImpl &&
            const DeepCollectionEquality().equals(other._options, _options));
  }

  @override
  int get hashCode =>
      Object.hash(runtimeType, const DeepCollectionEquality().hash(_options));

  /// Create a copy of FfiDecisionQuestionKind
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiDecisionQuestionKind_ChoiceImplCopyWith<
          _$FfiDecisionQuestionKind_ChoiceImpl>
      get copyWith => __$$FfiDecisionQuestionKind_ChoiceImplCopyWithImpl<
          _$FfiDecisionQuestionKind_ChoiceImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(List<FfiDecisionOption> options) choice,
    required TResult Function(FfiDecisionContent? yes, FfiDecisionContent? no)
        yesNo,
    required TResult Function(List<FfiDecisionLevel> levels) score,
  }) {
    return choice(options);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(List<FfiDecisionOption> options)? choice,
    TResult? Function(FfiDecisionContent? yes, FfiDecisionContent? no)? yesNo,
    TResult? Function(List<FfiDecisionLevel> levels)? score,
  }) {
    return choice?.call(options);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(List<FfiDecisionOption> options)? choice,
    TResult Function(FfiDecisionContent? yes, FfiDecisionContent? no)? yesNo,
    TResult Function(List<FfiDecisionLevel> levels)? score,
    required TResult orElse(),
  }) {
    if (choice != null) {
      return choice(options);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionQuestionKind_Choice value) choice,
    required TResult Function(FfiDecisionQuestionKind_YesNo value) yesNo,
    required TResult Function(FfiDecisionQuestionKind_Score value) score,
  }) {
    return choice(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionQuestionKind_Choice value)? choice,
    TResult? Function(FfiDecisionQuestionKind_YesNo value)? yesNo,
    TResult? Function(FfiDecisionQuestionKind_Score value)? score,
  }) {
    return choice?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionQuestionKind_Choice value)? choice,
    TResult Function(FfiDecisionQuestionKind_YesNo value)? yesNo,
    TResult Function(FfiDecisionQuestionKind_Score value)? score,
    required TResult orElse(),
  }) {
    if (choice != null) {
      return choice(this);
    }
    return orElse();
  }
}

abstract class FfiDecisionQuestionKind_Choice extends FfiDecisionQuestionKind {
  const factory FfiDecisionQuestionKind_Choice(
          {required final List<FfiDecisionOption> options}) =
      _$FfiDecisionQuestionKind_ChoiceImpl;
  const FfiDecisionQuestionKind_Choice._() : super._();

  List<FfiDecisionOption> get options;

  /// Create a copy of FfiDecisionQuestionKind
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiDecisionQuestionKind_ChoiceImplCopyWith<
          _$FfiDecisionQuestionKind_ChoiceImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiDecisionQuestionKind_YesNoImplCopyWith<$Res> {
  factory _$$FfiDecisionQuestionKind_YesNoImplCopyWith(
          _$FfiDecisionQuestionKind_YesNoImpl value,
          $Res Function(_$FfiDecisionQuestionKind_YesNoImpl) then) =
      __$$FfiDecisionQuestionKind_YesNoImplCopyWithImpl<$Res>;
  @useResult
  $Res call({FfiDecisionContent? yes, FfiDecisionContent? no});

  $FfiDecisionContentCopyWith<$Res>? get yes;
  $FfiDecisionContentCopyWith<$Res>? get no;
}

/// @nodoc
class __$$FfiDecisionQuestionKind_YesNoImplCopyWithImpl<$Res>
    extends _$FfiDecisionQuestionKindCopyWithImpl<$Res,
        _$FfiDecisionQuestionKind_YesNoImpl>
    implements _$$FfiDecisionQuestionKind_YesNoImplCopyWith<$Res> {
  __$$FfiDecisionQuestionKind_YesNoImplCopyWithImpl(
      _$FfiDecisionQuestionKind_YesNoImpl _value,
      $Res Function(_$FfiDecisionQuestionKind_YesNoImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiDecisionQuestionKind
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? yes = freezed,
    Object? no = freezed,
  }) {
    return _then(_$FfiDecisionQuestionKind_YesNoImpl(
      yes: freezed == yes
          ? _value.yes
          : yes // ignore: cast_nullable_to_non_nullable
              as FfiDecisionContent?,
      no: freezed == no
          ? _value.no
          : no // ignore: cast_nullable_to_non_nullable
              as FfiDecisionContent?,
    ));
  }

  /// Create a copy of FfiDecisionQuestionKind
  /// with the given fields replaced by the non-null parameter values.
  @override
  @pragma('vm:prefer-inline')
  $FfiDecisionContentCopyWith<$Res>? get yes {
    if (_value.yes == null) {
      return null;
    }

    return $FfiDecisionContentCopyWith<$Res>(_value.yes!, (value) {
      return _then(_value.copyWith(yes: value));
    });
  }

  /// Create a copy of FfiDecisionQuestionKind
  /// with the given fields replaced by the non-null parameter values.
  @override
  @pragma('vm:prefer-inline')
  $FfiDecisionContentCopyWith<$Res>? get no {
    if (_value.no == null) {
      return null;
    }

    return $FfiDecisionContentCopyWith<$Res>(_value.no!, (value) {
      return _then(_value.copyWith(no: value));
    });
  }
}

/// @nodoc

class _$FfiDecisionQuestionKind_YesNoImpl
    extends FfiDecisionQuestionKind_YesNo {
  const _$FfiDecisionQuestionKind_YesNoImpl({this.yes, this.no}) : super._();

  @override
  final FfiDecisionContent? yes;
  @override
  final FfiDecisionContent? no;

  @override
  String toString() {
    return 'FfiDecisionQuestionKind.yesNo(yes: $yes, no: $no)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiDecisionQuestionKind_YesNoImpl &&
            (identical(other.yes, yes) || other.yes == yes) &&
            (identical(other.no, no) || other.no == no));
  }

  @override
  int get hashCode => Object.hash(runtimeType, yes, no);

  /// Create a copy of FfiDecisionQuestionKind
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiDecisionQuestionKind_YesNoImplCopyWith<
          _$FfiDecisionQuestionKind_YesNoImpl>
      get copyWith => __$$FfiDecisionQuestionKind_YesNoImplCopyWithImpl<
          _$FfiDecisionQuestionKind_YesNoImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(List<FfiDecisionOption> options) choice,
    required TResult Function(FfiDecisionContent? yes, FfiDecisionContent? no)
        yesNo,
    required TResult Function(List<FfiDecisionLevel> levels) score,
  }) {
    return yesNo(yes, no);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(List<FfiDecisionOption> options)? choice,
    TResult? Function(FfiDecisionContent? yes, FfiDecisionContent? no)? yesNo,
    TResult? Function(List<FfiDecisionLevel> levels)? score,
  }) {
    return yesNo?.call(yes, no);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(List<FfiDecisionOption> options)? choice,
    TResult Function(FfiDecisionContent? yes, FfiDecisionContent? no)? yesNo,
    TResult Function(List<FfiDecisionLevel> levels)? score,
    required TResult orElse(),
  }) {
    if (yesNo != null) {
      return yesNo(yes, no);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionQuestionKind_Choice value) choice,
    required TResult Function(FfiDecisionQuestionKind_YesNo value) yesNo,
    required TResult Function(FfiDecisionQuestionKind_Score value) score,
  }) {
    return yesNo(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionQuestionKind_Choice value)? choice,
    TResult? Function(FfiDecisionQuestionKind_YesNo value)? yesNo,
    TResult? Function(FfiDecisionQuestionKind_Score value)? score,
  }) {
    return yesNo?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionQuestionKind_Choice value)? choice,
    TResult Function(FfiDecisionQuestionKind_YesNo value)? yesNo,
    TResult Function(FfiDecisionQuestionKind_Score value)? score,
    required TResult orElse(),
  }) {
    if (yesNo != null) {
      return yesNo(this);
    }
    return orElse();
  }
}

abstract class FfiDecisionQuestionKind_YesNo extends FfiDecisionQuestionKind {
  const factory FfiDecisionQuestionKind_YesNo(
      {final FfiDecisionContent? yes,
      final FfiDecisionContent? no}) = _$FfiDecisionQuestionKind_YesNoImpl;
  const FfiDecisionQuestionKind_YesNo._() : super._();

  FfiDecisionContent? get yes;
  FfiDecisionContent? get no;

  /// Create a copy of FfiDecisionQuestionKind
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiDecisionQuestionKind_YesNoImplCopyWith<
          _$FfiDecisionQuestionKind_YesNoImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiDecisionQuestionKind_ScoreImplCopyWith<$Res> {
  factory _$$FfiDecisionQuestionKind_ScoreImplCopyWith(
          _$FfiDecisionQuestionKind_ScoreImpl value,
          $Res Function(_$FfiDecisionQuestionKind_ScoreImpl) then) =
      __$$FfiDecisionQuestionKind_ScoreImplCopyWithImpl<$Res>;
  @useResult
  $Res call({List<FfiDecisionLevel> levels});
}

/// @nodoc
class __$$FfiDecisionQuestionKind_ScoreImplCopyWithImpl<$Res>
    extends _$FfiDecisionQuestionKindCopyWithImpl<$Res,
        _$FfiDecisionQuestionKind_ScoreImpl>
    implements _$$FfiDecisionQuestionKind_ScoreImplCopyWith<$Res> {
  __$$FfiDecisionQuestionKind_ScoreImplCopyWithImpl(
      _$FfiDecisionQuestionKind_ScoreImpl _value,
      $Res Function(_$FfiDecisionQuestionKind_ScoreImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiDecisionQuestionKind
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? levels = null,
  }) {
    return _then(_$FfiDecisionQuestionKind_ScoreImpl(
      levels: null == levels
          ? _value._levels
          : levels // ignore: cast_nullable_to_non_nullable
              as List<FfiDecisionLevel>,
    ));
  }
}

/// @nodoc

class _$FfiDecisionQuestionKind_ScoreImpl
    extends FfiDecisionQuestionKind_Score {
  const _$FfiDecisionQuestionKind_ScoreImpl(
      {required final List<FfiDecisionLevel> levels})
      : _levels = levels,
        super._();

  final List<FfiDecisionLevel> _levels;
  @override
  List<FfiDecisionLevel> get levels {
    if (_levels is EqualUnmodifiableListView) return _levels;
    // ignore: implicit_dynamic_type
    return EqualUnmodifiableListView(_levels);
  }

  @override
  String toString() {
    return 'FfiDecisionQuestionKind.score(levels: $levels)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiDecisionQuestionKind_ScoreImpl &&
            const DeepCollectionEquality().equals(other._levels, _levels));
  }

  @override
  int get hashCode =>
      Object.hash(runtimeType, const DeepCollectionEquality().hash(_levels));

  /// Create a copy of FfiDecisionQuestionKind
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiDecisionQuestionKind_ScoreImplCopyWith<
          _$FfiDecisionQuestionKind_ScoreImpl>
      get copyWith => __$$FfiDecisionQuestionKind_ScoreImplCopyWithImpl<
          _$FfiDecisionQuestionKind_ScoreImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(List<FfiDecisionOption> options) choice,
    required TResult Function(FfiDecisionContent? yes, FfiDecisionContent? no)
        yesNo,
    required TResult Function(List<FfiDecisionLevel> levels) score,
  }) {
    return score(levels);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(List<FfiDecisionOption> options)? choice,
    TResult? Function(FfiDecisionContent? yes, FfiDecisionContent? no)? yesNo,
    TResult? Function(List<FfiDecisionLevel> levels)? score,
  }) {
    return score?.call(levels);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(List<FfiDecisionOption> options)? choice,
    TResult Function(FfiDecisionContent? yes, FfiDecisionContent? no)? yesNo,
    TResult Function(List<FfiDecisionLevel> levels)? score,
    required TResult orElse(),
  }) {
    if (score != null) {
      return score(levels);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionQuestionKind_Choice value) choice,
    required TResult Function(FfiDecisionQuestionKind_YesNo value) yesNo,
    required TResult Function(FfiDecisionQuestionKind_Score value) score,
  }) {
    return score(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionQuestionKind_Choice value)? choice,
    TResult? Function(FfiDecisionQuestionKind_YesNo value)? yesNo,
    TResult? Function(FfiDecisionQuestionKind_Score value)? score,
  }) {
    return score?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionQuestionKind_Choice value)? choice,
    TResult Function(FfiDecisionQuestionKind_YesNo value)? yesNo,
    TResult Function(FfiDecisionQuestionKind_Score value)? score,
    required TResult orElse(),
  }) {
    if (score != null) {
      return score(this);
    }
    return orElse();
  }
}

abstract class FfiDecisionQuestionKind_Score extends FfiDecisionQuestionKind {
  const factory FfiDecisionQuestionKind_Score(
          {required final List<FfiDecisionLevel> levels}) =
      _$FfiDecisionQuestionKind_ScoreImpl;
  const FfiDecisionQuestionKind_Score._() : super._();

  List<FfiDecisionLevel> get levels;

  /// Create a copy of FfiDecisionQuestionKind
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiDecisionQuestionKind_ScoreImplCopyWith<
          _$FfiDecisionQuestionKind_ScoreImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
mixin _$FfiDecisionValue {
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String key) choice,
    required TResult Function(double pYes) yesNo,
    required TResult Function(double value, FfiScoreStatistic statistic) score,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String key)? choice,
    TResult? Function(double pYes)? yesNo,
    TResult? Function(double value, FfiScoreStatistic statistic)? score,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String key)? choice,
    TResult Function(double pYes)? yesNo,
    TResult Function(double value, FfiScoreStatistic statistic)? score,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionValue_Choice value) choice,
    required TResult Function(FfiDecisionValue_YesNo value) yesNo,
    required TResult Function(FfiDecisionValue_Score value) score,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionValue_Choice value)? choice,
    TResult? Function(FfiDecisionValue_YesNo value)? yesNo,
    TResult? Function(FfiDecisionValue_Score value)? score,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionValue_Choice value)? choice,
    TResult Function(FfiDecisionValue_YesNo value)? yesNo,
    TResult Function(FfiDecisionValue_Score value)? score,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $FfiDecisionValueCopyWith<$Res> {
  factory $FfiDecisionValueCopyWith(
          FfiDecisionValue value, $Res Function(FfiDecisionValue) then) =
      _$FfiDecisionValueCopyWithImpl<$Res, FfiDecisionValue>;
}

/// @nodoc
class _$FfiDecisionValueCopyWithImpl<$Res, $Val extends FfiDecisionValue>
    implements $FfiDecisionValueCopyWith<$Res> {
  _$FfiDecisionValueCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
}

/// @nodoc
abstract class _$$FfiDecisionValue_ChoiceImplCopyWith<$Res> {
  factory _$$FfiDecisionValue_ChoiceImplCopyWith(
          _$FfiDecisionValue_ChoiceImpl value,
          $Res Function(_$FfiDecisionValue_ChoiceImpl) then) =
      __$$FfiDecisionValue_ChoiceImplCopyWithImpl<$Res>;
  @useResult
  $Res call({String key});
}

/// @nodoc
class __$$FfiDecisionValue_ChoiceImplCopyWithImpl<$Res>
    extends _$FfiDecisionValueCopyWithImpl<$Res, _$FfiDecisionValue_ChoiceImpl>
    implements _$$FfiDecisionValue_ChoiceImplCopyWith<$Res> {
  __$$FfiDecisionValue_ChoiceImplCopyWithImpl(
      _$FfiDecisionValue_ChoiceImpl _value,
      $Res Function(_$FfiDecisionValue_ChoiceImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? key = null,
  }) {
    return _then(_$FfiDecisionValue_ChoiceImpl(
      key: null == key
          ? _value.key
          : key // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class _$FfiDecisionValue_ChoiceImpl extends FfiDecisionValue_Choice {
  const _$FfiDecisionValue_ChoiceImpl({required this.key}) : super._();

  @override
  final String key;

  @override
  String toString() {
    return 'FfiDecisionValue.choice(key: $key)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiDecisionValue_ChoiceImpl &&
            (identical(other.key, key) || other.key == key));
  }

  @override
  int get hashCode => Object.hash(runtimeType, key);

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiDecisionValue_ChoiceImplCopyWith<_$FfiDecisionValue_ChoiceImpl>
      get copyWith => __$$FfiDecisionValue_ChoiceImplCopyWithImpl<
          _$FfiDecisionValue_ChoiceImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String key) choice,
    required TResult Function(double pYes) yesNo,
    required TResult Function(double value, FfiScoreStatistic statistic) score,
  }) {
    return choice(key);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String key)? choice,
    TResult? Function(double pYes)? yesNo,
    TResult? Function(double value, FfiScoreStatistic statistic)? score,
  }) {
    return choice?.call(key);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String key)? choice,
    TResult Function(double pYes)? yesNo,
    TResult Function(double value, FfiScoreStatistic statistic)? score,
    required TResult orElse(),
  }) {
    if (choice != null) {
      return choice(key);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionValue_Choice value) choice,
    required TResult Function(FfiDecisionValue_YesNo value) yesNo,
    required TResult Function(FfiDecisionValue_Score value) score,
  }) {
    return choice(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionValue_Choice value)? choice,
    TResult? Function(FfiDecisionValue_YesNo value)? yesNo,
    TResult? Function(FfiDecisionValue_Score value)? score,
  }) {
    return choice?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionValue_Choice value)? choice,
    TResult Function(FfiDecisionValue_YesNo value)? yesNo,
    TResult Function(FfiDecisionValue_Score value)? score,
    required TResult orElse(),
  }) {
    if (choice != null) {
      return choice(this);
    }
    return orElse();
  }
}

abstract class FfiDecisionValue_Choice extends FfiDecisionValue {
  const factory FfiDecisionValue_Choice({required final String key}) =
      _$FfiDecisionValue_ChoiceImpl;
  const FfiDecisionValue_Choice._() : super._();

  String get key;

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiDecisionValue_ChoiceImplCopyWith<_$FfiDecisionValue_ChoiceImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiDecisionValue_YesNoImplCopyWith<$Res> {
  factory _$$FfiDecisionValue_YesNoImplCopyWith(
          _$FfiDecisionValue_YesNoImpl value,
          $Res Function(_$FfiDecisionValue_YesNoImpl) then) =
      __$$FfiDecisionValue_YesNoImplCopyWithImpl<$Res>;
  @useResult
  $Res call({double pYes});
}

/// @nodoc
class __$$FfiDecisionValue_YesNoImplCopyWithImpl<$Res>
    extends _$FfiDecisionValueCopyWithImpl<$Res, _$FfiDecisionValue_YesNoImpl>
    implements _$$FfiDecisionValue_YesNoImplCopyWith<$Res> {
  __$$FfiDecisionValue_YesNoImplCopyWithImpl(
      _$FfiDecisionValue_YesNoImpl _value,
      $Res Function(_$FfiDecisionValue_YesNoImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? pYes = null,
  }) {
    return _then(_$FfiDecisionValue_YesNoImpl(
      pYes: null == pYes
          ? _value.pYes
          : pYes // ignore: cast_nullable_to_non_nullable
              as double,
    ));
  }
}

/// @nodoc

class _$FfiDecisionValue_YesNoImpl extends FfiDecisionValue_YesNo {
  const _$FfiDecisionValue_YesNoImpl({required this.pYes}) : super._();

  @override
  final double pYes;

  @override
  String toString() {
    return 'FfiDecisionValue.yesNo(pYes: $pYes)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiDecisionValue_YesNoImpl &&
            (identical(other.pYes, pYes) || other.pYes == pYes));
  }

  @override
  int get hashCode => Object.hash(runtimeType, pYes);

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiDecisionValue_YesNoImplCopyWith<_$FfiDecisionValue_YesNoImpl>
      get copyWith => __$$FfiDecisionValue_YesNoImplCopyWithImpl<
          _$FfiDecisionValue_YesNoImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String key) choice,
    required TResult Function(double pYes) yesNo,
    required TResult Function(double value, FfiScoreStatistic statistic) score,
  }) {
    return yesNo(pYes);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String key)? choice,
    TResult? Function(double pYes)? yesNo,
    TResult? Function(double value, FfiScoreStatistic statistic)? score,
  }) {
    return yesNo?.call(pYes);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String key)? choice,
    TResult Function(double pYes)? yesNo,
    TResult Function(double value, FfiScoreStatistic statistic)? score,
    required TResult orElse(),
  }) {
    if (yesNo != null) {
      return yesNo(pYes);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionValue_Choice value) choice,
    required TResult Function(FfiDecisionValue_YesNo value) yesNo,
    required TResult Function(FfiDecisionValue_Score value) score,
  }) {
    return yesNo(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionValue_Choice value)? choice,
    TResult? Function(FfiDecisionValue_YesNo value)? yesNo,
    TResult? Function(FfiDecisionValue_Score value)? score,
  }) {
    return yesNo?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionValue_Choice value)? choice,
    TResult Function(FfiDecisionValue_YesNo value)? yesNo,
    TResult Function(FfiDecisionValue_Score value)? score,
    required TResult orElse(),
  }) {
    if (yesNo != null) {
      return yesNo(this);
    }
    return orElse();
  }
}

abstract class FfiDecisionValue_YesNo extends FfiDecisionValue {
  const factory FfiDecisionValue_YesNo({required final double pYes}) =
      _$FfiDecisionValue_YesNoImpl;
  const FfiDecisionValue_YesNo._() : super._();

  double get pYes;

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiDecisionValue_YesNoImplCopyWith<_$FfiDecisionValue_YesNoImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiDecisionValue_ScoreImplCopyWith<$Res> {
  factory _$$FfiDecisionValue_ScoreImplCopyWith(
          _$FfiDecisionValue_ScoreImpl value,
          $Res Function(_$FfiDecisionValue_ScoreImpl) then) =
      __$$FfiDecisionValue_ScoreImplCopyWithImpl<$Res>;
  @useResult
  $Res call({double value, FfiScoreStatistic statistic});
}

/// @nodoc
class __$$FfiDecisionValue_ScoreImplCopyWithImpl<$Res>
    extends _$FfiDecisionValueCopyWithImpl<$Res, _$FfiDecisionValue_ScoreImpl>
    implements _$$FfiDecisionValue_ScoreImplCopyWith<$Res> {
  __$$FfiDecisionValue_ScoreImplCopyWithImpl(
      _$FfiDecisionValue_ScoreImpl _value,
      $Res Function(_$FfiDecisionValue_ScoreImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? value = null,
    Object? statistic = null,
  }) {
    return _then(_$FfiDecisionValue_ScoreImpl(
      value: null == value
          ? _value.value
          : value // ignore: cast_nullable_to_non_nullable
              as double,
      statistic: null == statistic
          ? _value.statistic
          : statistic // ignore: cast_nullable_to_non_nullable
              as FfiScoreStatistic,
    ));
  }
}

/// @nodoc

class _$FfiDecisionValue_ScoreImpl extends FfiDecisionValue_Score {
  const _$FfiDecisionValue_ScoreImpl(
      {required this.value, required this.statistic})
      : super._();

  @override
  final double value;
  @override
  final FfiScoreStatistic statistic;

  @override
  String toString() {
    return 'FfiDecisionValue.score(value: $value, statistic: $statistic)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiDecisionValue_ScoreImpl &&
            (identical(other.value, value) || other.value == value) &&
            (identical(other.statistic, statistic) ||
                other.statistic == statistic));
  }

  @override
  int get hashCode => Object.hash(runtimeType, value, statistic);

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiDecisionValue_ScoreImplCopyWith<_$FfiDecisionValue_ScoreImpl>
      get copyWith => __$$FfiDecisionValue_ScoreImplCopyWithImpl<
          _$FfiDecisionValue_ScoreImpl>(this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String key) choice,
    required TResult Function(double pYes) yesNo,
    required TResult Function(double value, FfiScoreStatistic statistic) score,
  }) {
    return score(value, statistic);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String key)? choice,
    TResult? Function(double pYes)? yesNo,
    TResult? Function(double value, FfiScoreStatistic statistic)? score,
  }) {
    return score?.call(value, statistic);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String key)? choice,
    TResult Function(double pYes)? yesNo,
    TResult Function(double value, FfiScoreStatistic statistic)? score,
    required TResult orElse(),
  }) {
    if (score != null) {
      return score(value, statistic);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiDecisionValue_Choice value) choice,
    required TResult Function(FfiDecisionValue_YesNo value) yesNo,
    required TResult Function(FfiDecisionValue_Score value) score,
  }) {
    return score(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiDecisionValue_Choice value)? choice,
    TResult? Function(FfiDecisionValue_YesNo value)? yesNo,
    TResult? Function(FfiDecisionValue_Score value)? score,
  }) {
    return score?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiDecisionValue_Choice value)? choice,
    TResult Function(FfiDecisionValue_YesNo value)? yesNo,
    TResult Function(FfiDecisionValue_Score value)? score,
    required TResult orElse(),
  }) {
    if (score != null) {
      return score(this);
    }
    return orElse();
  }
}

abstract class FfiDecisionValue_Score extends FfiDecisionValue {
  const factory FfiDecisionValue_Score(
          {required final double value,
          required final FfiScoreStatistic statistic}) =
      _$FfiDecisionValue_ScoreImpl;
  const FfiDecisionValue_Score._() : super._();

  double get value;
  FfiScoreStatistic get statistic;

  /// Create a copy of FfiDecisionValue
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiDecisionValue_ScoreImplCopyWith<_$FfiDecisionValue_ScoreImpl>
      get copyWith => throw _privateConstructorUsedError;
}

/// @nodoc
mixin _$FfiProbeJson {
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function() null_,
    required TResult Function(bool value) bool,
    required TResult Function(int value) int,
    required TResult Function(BigInt value) uint,
    required TResult Function(double value) float,
    required TResult Function(String value) text,
    required TResult Function(List<FfiProbeJson> items) seq,
    required TResult Function(List<FfiProbeJsonEntry> entries) obj,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function()? null_,
    TResult? Function(bool value)? bool,
    TResult? Function(int value)? int,
    TResult? Function(BigInt value)? uint,
    TResult? Function(double value)? float,
    TResult? Function(String value)? text,
    TResult? Function(List<FfiProbeJson> items)? seq,
    TResult? Function(List<FfiProbeJsonEntry> entries)? obj,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function()? null_,
    TResult Function(bool value)? bool,
    TResult Function(int value)? int,
    TResult Function(BigInt value)? uint,
    TResult Function(double value)? float,
    TResult Function(String value)? text,
    TResult Function(List<FfiProbeJson> items)? seq,
    TResult Function(List<FfiProbeJsonEntry> entries)? obj,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiProbeJson_Null value) null_,
    required TResult Function(FfiProbeJson_Bool value) bool,
    required TResult Function(FfiProbeJson_Int value) int,
    required TResult Function(FfiProbeJson_Uint value) uint,
    required TResult Function(FfiProbeJson_Float value) float,
    required TResult Function(FfiProbeJson_Text value) text,
    required TResult Function(FfiProbeJson_Seq value) seq,
    required TResult Function(FfiProbeJson_Obj value) obj,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiProbeJson_Null value)? null_,
    TResult? Function(FfiProbeJson_Bool value)? bool,
    TResult? Function(FfiProbeJson_Int value)? int,
    TResult? Function(FfiProbeJson_Uint value)? uint,
    TResult? Function(FfiProbeJson_Float value)? float,
    TResult? Function(FfiProbeJson_Text value)? text,
    TResult? Function(FfiProbeJson_Seq value)? seq,
    TResult? Function(FfiProbeJson_Obj value)? obj,
  }) =>
      throw _privateConstructorUsedError;
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiProbeJson_Null value)? null_,
    TResult Function(FfiProbeJson_Bool value)? bool,
    TResult Function(FfiProbeJson_Int value)? int,
    TResult Function(FfiProbeJson_Uint value)? uint,
    TResult Function(FfiProbeJson_Float value)? float,
    TResult Function(FfiProbeJson_Text value)? text,
    TResult Function(FfiProbeJson_Seq value)? seq,
    TResult Function(FfiProbeJson_Obj value)? obj,
    required TResult orElse(),
  }) =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $FfiProbeJsonCopyWith<$Res> {
  factory $FfiProbeJsonCopyWith(
          FfiProbeJson value, $Res Function(FfiProbeJson) then) =
      _$FfiProbeJsonCopyWithImpl<$Res, FfiProbeJson>;
}

/// @nodoc
class _$FfiProbeJsonCopyWithImpl<$Res, $Val extends FfiProbeJson>
    implements $FfiProbeJsonCopyWith<$Res> {
  _$FfiProbeJsonCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
}

/// @nodoc
abstract class _$$FfiProbeJson_NullImplCopyWith<$Res> {
  factory _$$FfiProbeJson_NullImplCopyWith(_$FfiProbeJson_NullImpl value,
          $Res Function(_$FfiProbeJson_NullImpl) then) =
      __$$FfiProbeJson_NullImplCopyWithImpl<$Res>;
}

/// @nodoc
class __$$FfiProbeJson_NullImplCopyWithImpl<$Res>
    extends _$FfiProbeJsonCopyWithImpl<$Res, _$FfiProbeJson_NullImpl>
    implements _$$FfiProbeJson_NullImplCopyWith<$Res> {
  __$$FfiProbeJson_NullImplCopyWithImpl(_$FfiProbeJson_NullImpl _value,
      $Res Function(_$FfiProbeJson_NullImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
}

/// @nodoc

class _$FfiProbeJson_NullImpl extends FfiProbeJson_Null {
  const _$FfiProbeJson_NullImpl() : super._();

  @override
  String toString() {
    return 'FfiProbeJson.null_()';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType && other is _$FfiProbeJson_NullImpl);
  }

  @override
  int get hashCode => runtimeType.hashCode;

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function() null_,
    required TResult Function(bool value) bool,
    required TResult Function(int value) int,
    required TResult Function(BigInt value) uint,
    required TResult Function(double value) float,
    required TResult Function(String value) text,
    required TResult Function(List<FfiProbeJson> items) seq,
    required TResult Function(List<FfiProbeJsonEntry> entries) obj,
  }) {
    return null_();
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function()? null_,
    TResult? Function(bool value)? bool,
    TResult? Function(int value)? int,
    TResult? Function(BigInt value)? uint,
    TResult? Function(double value)? float,
    TResult? Function(String value)? text,
    TResult? Function(List<FfiProbeJson> items)? seq,
    TResult? Function(List<FfiProbeJsonEntry> entries)? obj,
  }) {
    return null_?.call();
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function()? null_,
    TResult Function(bool value)? bool,
    TResult Function(int value)? int,
    TResult Function(BigInt value)? uint,
    TResult Function(double value)? float,
    TResult Function(String value)? text,
    TResult Function(List<FfiProbeJson> items)? seq,
    TResult Function(List<FfiProbeJsonEntry> entries)? obj,
    required TResult orElse(),
  }) {
    if (null_ != null) {
      return null_();
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiProbeJson_Null value) null_,
    required TResult Function(FfiProbeJson_Bool value) bool,
    required TResult Function(FfiProbeJson_Int value) int,
    required TResult Function(FfiProbeJson_Uint value) uint,
    required TResult Function(FfiProbeJson_Float value) float,
    required TResult Function(FfiProbeJson_Text value) text,
    required TResult Function(FfiProbeJson_Seq value) seq,
    required TResult Function(FfiProbeJson_Obj value) obj,
  }) {
    return null_(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiProbeJson_Null value)? null_,
    TResult? Function(FfiProbeJson_Bool value)? bool,
    TResult? Function(FfiProbeJson_Int value)? int,
    TResult? Function(FfiProbeJson_Uint value)? uint,
    TResult? Function(FfiProbeJson_Float value)? float,
    TResult? Function(FfiProbeJson_Text value)? text,
    TResult? Function(FfiProbeJson_Seq value)? seq,
    TResult? Function(FfiProbeJson_Obj value)? obj,
  }) {
    return null_?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiProbeJson_Null value)? null_,
    TResult Function(FfiProbeJson_Bool value)? bool,
    TResult Function(FfiProbeJson_Int value)? int,
    TResult Function(FfiProbeJson_Uint value)? uint,
    TResult Function(FfiProbeJson_Float value)? float,
    TResult Function(FfiProbeJson_Text value)? text,
    TResult Function(FfiProbeJson_Seq value)? seq,
    TResult Function(FfiProbeJson_Obj value)? obj,
    required TResult orElse(),
  }) {
    if (null_ != null) {
      return null_(this);
    }
    return orElse();
  }
}

abstract class FfiProbeJson_Null extends FfiProbeJson {
  const factory FfiProbeJson_Null() = _$FfiProbeJson_NullImpl;
  const FfiProbeJson_Null._() : super._();
}

/// @nodoc
abstract class _$$FfiProbeJson_BoolImplCopyWith<$Res> {
  factory _$$FfiProbeJson_BoolImplCopyWith(_$FfiProbeJson_BoolImpl value,
          $Res Function(_$FfiProbeJson_BoolImpl) then) =
      __$$FfiProbeJson_BoolImplCopyWithImpl<$Res>;
  @useResult
  $Res call({bool value});
}

/// @nodoc
class __$$FfiProbeJson_BoolImplCopyWithImpl<$Res>
    extends _$FfiProbeJsonCopyWithImpl<$Res, _$FfiProbeJson_BoolImpl>
    implements _$$FfiProbeJson_BoolImplCopyWith<$Res> {
  __$$FfiProbeJson_BoolImplCopyWithImpl(_$FfiProbeJson_BoolImpl _value,
      $Res Function(_$FfiProbeJson_BoolImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? value = null,
  }) {
    return _then(_$FfiProbeJson_BoolImpl(
      value: null == value
          ? _value.value
          : value // ignore: cast_nullable_to_non_nullable
              as bool,
    ));
  }
}

/// @nodoc

class _$FfiProbeJson_BoolImpl extends FfiProbeJson_Bool {
  const _$FfiProbeJson_BoolImpl({required this.value}) : super._();

  @override
  final bool value;

  @override
  String toString() {
    return 'FfiProbeJson.bool(value: $value)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiProbeJson_BoolImpl &&
            (identical(other.value, value) || other.value == value));
  }

  @override
  int get hashCode => Object.hash(runtimeType, value);

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiProbeJson_BoolImplCopyWith<_$FfiProbeJson_BoolImpl> get copyWith =>
      __$$FfiProbeJson_BoolImplCopyWithImpl<_$FfiProbeJson_BoolImpl>(
          this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function() null_,
    required TResult Function(bool value) bool,
    required TResult Function(int value) int,
    required TResult Function(BigInt value) uint,
    required TResult Function(double value) float,
    required TResult Function(String value) text,
    required TResult Function(List<FfiProbeJson> items) seq,
    required TResult Function(List<FfiProbeJsonEntry> entries) obj,
  }) {
    return bool(value);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function()? null_,
    TResult? Function(bool value)? bool,
    TResult? Function(int value)? int,
    TResult? Function(BigInt value)? uint,
    TResult? Function(double value)? float,
    TResult? Function(String value)? text,
    TResult? Function(List<FfiProbeJson> items)? seq,
    TResult? Function(List<FfiProbeJsonEntry> entries)? obj,
  }) {
    return bool?.call(value);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function()? null_,
    TResult Function(bool value)? bool,
    TResult Function(int value)? int,
    TResult Function(BigInt value)? uint,
    TResult Function(double value)? float,
    TResult Function(String value)? text,
    TResult Function(List<FfiProbeJson> items)? seq,
    TResult Function(List<FfiProbeJsonEntry> entries)? obj,
    required TResult orElse(),
  }) {
    if (bool != null) {
      return bool(value);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiProbeJson_Null value) null_,
    required TResult Function(FfiProbeJson_Bool value) bool,
    required TResult Function(FfiProbeJson_Int value) int,
    required TResult Function(FfiProbeJson_Uint value) uint,
    required TResult Function(FfiProbeJson_Float value) float,
    required TResult Function(FfiProbeJson_Text value) text,
    required TResult Function(FfiProbeJson_Seq value) seq,
    required TResult Function(FfiProbeJson_Obj value) obj,
  }) {
    return bool(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiProbeJson_Null value)? null_,
    TResult? Function(FfiProbeJson_Bool value)? bool,
    TResult? Function(FfiProbeJson_Int value)? int,
    TResult? Function(FfiProbeJson_Uint value)? uint,
    TResult? Function(FfiProbeJson_Float value)? float,
    TResult? Function(FfiProbeJson_Text value)? text,
    TResult? Function(FfiProbeJson_Seq value)? seq,
    TResult? Function(FfiProbeJson_Obj value)? obj,
  }) {
    return bool?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiProbeJson_Null value)? null_,
    TResult Function(FfiProbeJson_Bool value)? bool,
    TResult Function(FfiProbeJson_Int value)? int,
    TResult Function(FfiProbeJson_Uint value)? uint,
    TResult Function(FfiProbeJson_Float value)? float,
    TResult Function(FfiProbeJson_Text value)? text,
    TResult Function(FfiProbeJson_Seq value)? seq,
    TResult Function(FfiProbeJson_Obj value)? obj,
    required TResult orElse(),
  }) {
    if (bool != null) {
      return bool(this);
    }
    return orElse();
  }
}

abstract class FfiProbeJson_Bool extends FfiProbeJson {
  const factory FfiProbeJson_Bool({required final bool value}) =
      _$FfiProbeJson_BoolImpl;
  const FfiProbeJson_Bool._() : super._();

  bool get value;

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiProbeJson_BoolImplCopyWith<_$FfiProbeJson_BoolImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiProbeJson_IntImplCopyWith<$Res> {
  factory _$$FfiProbeJson_IntImplCopyWith(_$FfiProbeJson_IntImpl value,
          $Res Function(_$FfiProbeJson_IntImpl) then) =
      __$$FfiProbeJson_IntImplCopyWithImpl<$Res>;
  @useResult
  $Res call({int value});
}

/// @nodoc
class __$$FfiProbeJson_IntImplCopyWithImpl<$Res>
    extends _$FfiProbeJsonCopyWithImpl<$Res, _$FfiProbeJson_IntImpl>
    implements _$$FfiProbeJson_IntImplCopyWith<$Res> {
  __$$FfiProbeJson_IntImplCopyWithImpl(_$FfiProbeJson_IntImpl _value,
      $Res Function(_$FfiProbeJson_IntImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? value = null,
  }) {
    return _then(_$FfiProbeJson_IntImpl(
      value: null == value
          ? _value.value
          : value // ignore: cast_nullable_to_non_nullable
              as int,
    ));
  }
}

/// @nodoc

class _$FfiProbeJson_IntImpl extends FfiProbeJson_Int {
  const _$FfiProbeJson_IntImpl({required this.value}) : super._();

  @override
  final int value;

  @override
  String toString() {
    return 'FfiProbeJson.int(value: $value)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiProbeJson_IntImpl &&
            (identical(other.value, value) || other.value == value));
  }

  @override
  int get hashCode => Object.hash(runtimeType, value);

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiProbeJson_IntImplCopyWith<_$FfiProbeJson_IntImpl> get copyWith =>
      __$$FfiProbeJson_IntImplCopyWithImpl<_$FfiProbeJson_IntImpl>(
          this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function() null_,
    required TResult Function(bool value) bool,
    required TResult Function(int value) int,
    required TResult Function(BigInt value) uint,
    required TResult Function(double value) float,
    required TResult Function(String value) text,
    required TResult Function(List<FfiProbeJson> items) seq,
    required TResult Function(List<FfiProbeJsonEntry> entries) obj,
  }) {
    return int(value);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function()? null_,
    TResult? Function(bool value)? bool,
    TResult? Function(int value)? int,
    TResult? Function(BigInt value)? uint,
    TResult? Function(double value)? float,
    TResult? Function(String value)? text,
    TResult? Function(List<FfiProbeJson> items)? seq,
    TResult? Function(List<FfiProbeJsonEntry> entries)? obj,
  }) {
    return int?.call(value);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function()? null_,
    TResult Function(bool value)? bool,
    TResult Function(int value)? int,
    TResult Function(BigInt value)? uint,
    TResult Function(double value)? float,
    TResult Function(String value)? text,
    TResult Function(List<FfiProbeJson> items)? seq,
    TResult Function(List<FfiProbeJsonEntry> entries)? obj,
    required TResult orElse(),
  }) {
    if (int != null) {
      return int(value);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiProbeJson_Null value) null_,
    required TResult Function(FfiProbeJson_Bool value) bool,
    required TResult Function(FfiProbeJson_Int value) int,
    required TResult Function(FfiProbeJson_Uint value) uint,
    required TResult Function(FfiProbeJson_Float value) float,
    required TResult Function(FfiProbeJson_Text value) text,
    required TResult Function(FfiProbeJson_Seq value) seq,
    required TResult Function(FfiProbeJson_Obj value) obj,
  }) {
    return int(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiProbeJson_Null value)? null_,
    TResult? Function(FfiProbeJson_Bool value)? bool,
    TResult? Function(FfiProbeJson_Int value)? int,
    TResult? Function(FfiProbeJson_Uint value)? uint,
    TResult? Function(FfiProbeJson_Float value)? float,
    TResult? Function(FfiProbeJson_Text value)? text,
    TResult? Function(FfiProbeJson_Seq value)? seq,
    TResult? Function(FfiProbeJson_Obj value)? obj,
  }) {
    return int?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiProbeJson_Null value)? null_,
    TResult Function(FfiProbeJson_Bool value)? bool,
    TResult Function(FfiProbeJson_Int value)? int,
    TResult Function(FfiProbeJson_Uint value)? uint,
    TResult Function(FfiProbeJson_Float value)? float,
    TResult Function(FfiProbeJson_Text value)? text,
    TResult Function(FfiProbeJson_Seq value)? seq,
    TResult Function(FfiProbeJson_Obj value)? obj,
    required TResult orElse(),
  }) {
    if (int != null) {
      return int(this);
    }
    return orElse();
  }
}

abstract class FfiProbeJson_Int extends FfiProbeJson {
  const factory FfiProbeJson_Int({required final int value}) =
      _$FfiProbeJson_IntImpl;
  const FfiProbeJson_Int._() : super._();

  int get value;

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiProbeJson_IntImplCopyWith<_$FfiProbeJson_IntImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiProbeJson_UintImplCopyWith<$Res> {
  factory _$$FfiProbeJson_UintImplCopyWith(_$FfiProbeJson_UintImpl value,
          $Res Function(_$FfiProbeJson_UintImpl) then) =
      __$$FfiProbeJson_UintImplCopyWithImpl<$Res>;
  @useResult
  $Res call({BigInt value});
}

/// @nodoc
class __$$FfiProbeJson_UintImplCopyWithImpl<$Res>
    extends _$FfiProbeJsonCopyWithImpl<$Res, _$FfiProbeJson_UintImpl>
    implements _$$FfiProbeJson_UintImplCopyWith<$Res> {
  __$$FfiProbeJson_UintImplCopyWithImpl(_$FfiProbeJson_UintImpl _value,
      $Res Function(_$FfiProbeJson_UintImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? value = null,
  }) {
    return _then(_$FfiProbeJson_UintImpl(
      value: null == value
          ? _value.value
          : value // ignore: cast_nullable_to_non_nullable
              as BigInt,
    ));
  }
}

/// @nodoc

class _$FfiProbeJson_UintImpl extends FfiProbeJson_Uint {
  const _$FfiProbeJson_UintImpl({required this.value}) : super._();

  @override
  final BigInt value;

  @override
  String toString() {
    return 'FfiProbeJson.uint(value: $value)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiProbeJson_UintImpl &&
            (identical(other.value, value) || other.value == value));
  }

  @override
  int get hashCode => Object.hash(runtimeType, value);

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiProbeJson_UintImplCopyWith<_$FfiProbeJson_UintImpl> get copyWith =>
      __$$FfiProbeJson_UintImplCopyWithImpl<_$FfiProbeJson_UintImpl>(
          this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function() null_,
    required TResult Function(bool value) bool,
    required TResult Function(int value) int,
    required TResult Function(BigInt value) uint,
    required TResult Function(double value) float,
    required TResult Function(String value) text,
    required TResult Function(List<FfiProbeJson> items) seq,
    required TResult Function(List<FfiProbeJsonEntry> entries) obj,
  }) {
    return uint(value);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function()? null_,
    TResult? Function(bool value)? bool,
    TResult? Function(int value)? int,
    TResult? Function(BigInt value)? uint,
    TResult? Function(double value)? float,
    TResult? Function(String value)? text,
    TResult? Function(List<FfiProbeJson> items)? seq,
    TResult? Function(List<FfiProbeJsonEntry> entries)? obj,
  }) {
    return uint?.call(value);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function()? null_,
    TResult Function(bool value)? bool,
    TResult Function(int value)? int,
    TResult Function(BigInt value)? uint,
    TResult Function(double value)? float,
    TResult Function(String value)? text,
    TResult Function(List<FfiProbeJson> items)? seq,
    TResult Function(List<FfiProbeJsonEntry> entries)? obj,
    required TResult orElse(),
  }) {
    if (uint != null) {
      return uint(value);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiProbeJson_Null value) null_,
    required TResult Function(FfiProbeJson_Bool value) bool,
    required TResult Function(FfiProbeJson_Int value) int,
    required TResult Function(FfiProbeJson_Uint value) uint,
    required TResult Function(FfiProbeJson_Float value) float,
    required TResult Function(FfiProbeJson_Text value) text,
    required TResult Function(FfiProbeJson_Seq value) seq,
    required TResult Function(FfiProbeJson_Obj value) obj,
  }) {
    return uint(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiProbeJson_Null value)? null_,
    TResult? Function(FfiProbeJson_Bool value)? bool,
    TResult? Function(FfiProbeJson_Int value)? int,
    TResult? Function(FfiProbeJson_Uint value)? uint,
    TResult? Function(FfiProbeJson_Float value)? float,
    TResult? Function(FfiProbeJson_Text value)? text,
    TResult? Function(FfiProbeJson_Seq value)? seq,
    TResult? Function(FfiProbeJson_Obj value)? obj,
  }) {
    return uint?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiProbeJson_Null value)? null_,
    TResult Function(FfiProbeJson_Bool value)? bool,
    TResult Function(FfiProbeJson_Int value)? int,
    TResult Function(FfiProbeJson_Uint value)? uint,
    TResult Function(FfiProbeJson_Float value)? float,
    TResult Function(FfiProbeJson_Text value)? text,
    TResult Function(FfiProbeJson_Seq value)? seq,
    TResult Function(FfiProbeJson_Obj value)? obj,
    required TResult orElse(),
  }) {
    if (uint != null) {
      return uint(this);
    }
    return orElse();
  }
}

abstract class FfiProbeJson_Uint extends FfiProbeJson {
  const factory FfiProbeJson_Uint({required final BigInt value}) =
      _$FfiProbeJson_UintImpl;
  const FfiProbeJson_Uint._() : super._();

  BigInt get value;

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiProbeJson_UintImplCopyWith<_$FfiProbeJson_UintImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiProbeJson_FloatImplCopyWith<$Res> {
  factory _$$FfiProbeJson_FloatImplCopyWith(_$FfiProbeJson_FloatImpl value,
          $Res Function(_$FfiProbeJson_FloatImpl) then) =
      __$$FfiProbeJson_FloatImplCopyWithImpl<$Res>;
  @useResult
  $Res call({double value});
}

/// @nodoc
class __$$FfiProbeJson_FloatImplCopyWithImpl<$Res>
    extends _$FfiProbeJsonCopyWithImpl<$Res, _$FfiProbeJson_FloatImpl>
    implements _$$FfiProbeJson_FloatImplCopyWith<$Res> {
  __$$FfiProbeJson_FloatImplCopyWithImpl(_$FfiProbeJson_FloatImpl _value,
      $Res Function(_$FfiProbeJson_FloatImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? value = null,
  }) {
    return _then(_$FfiProbeJson_FloatImpl(
      value: null == value
          ? _value.value
          : value // ignore: cast_nullable_to_non_nullable
              as double,
    ));
  }
}

/// @nodoc

class _$FfiProbeJson_FloatImpl extends FfiProbeJson_Float {
  const _$FfiProbeJson_FloatImpl({required this.value}) : super._();

  @override
  final double value;

  @override
  String toString() {
    return 'FfiProbeJson.float(value: $value)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiProbeJson_FloatImpl &&
            (identical(other.value, value) || other.value == value));
  }

  @override
  int get hashCode => Object.hash(runtimeType, value);

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiProbeJson_FloatImplCopyWith<_$FfiProbeJson_FloatImpl> get copyWith =>
      __$$FfiProbeJson_FloatImplCopyWithImpl<_$FfiProbeJson_FloatImpl>(
          this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function() null_,
    required TResult Function(bool value) bool,
    required TResult Function(int value) int,
    required TResult Function(BigInt value) uint,
    required TResult Function(double value) float,
    required TResult Function(String value) text,
    required TResult Function(List<FfiProbeJson> items) seq,
    required TResult Function(List<FfiProbeJsonEntry> entries) obj,
  }) {
    return float(value);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function()? null_,
    TResult? Function(bool value)? bool,
    TResult? Function(int value)? int,
    TResult? Function(BigInt value)? uint,
    TResult? Function(double value)? float,
    TResult? Function(String value)? text,
    TResult? Function(List<FfiProbeJson> items)? seq,
    TResult? Function(List<FfiProbeJsonEntry> entries)? obj,
  }) {
    return float?.call(value);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function()? null_,
    TResult Function(bool value)? bool,
    TResult Function(int value)? int,
    TResult Function(BigInt value)? uint,
    TResult Function(double value)? float,
    TResult Function(String value)? text,
    TResult Function(List<FfiProbeJson> items)? seq,
    TResult Function(List<FfiProbeJsonEntry> entries)? obj,
    required TResult orElse(),
  }) {
    if (float != null) {
      return float(value);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiProbeJson_Null value) null_,
    required TResult Function(FfiProbeJson_Bool value) bool,
    required TResult Function(FfiProbeJson_Int value) int,
    required TResult Function(FfiProbeJson_Uint value) uint,
    required TResult Function(FfiProbeJson_Float value) float,
    required TResult Function(FfiProbeJson_Text value) text,
    required TResult Function(FfiProbeJson_Seq value) seq,
    required TResult Function(FfiProbeJson_Obj value) obj,
  }) {
    return float(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiProbeJson_Null value)? null_,
    TResult? Function(FfiProbeJson_Bool value)? bool,
    TResult? Function(FfiProbeJson_Int value)? int,
    TResult? Function(FfiProbeJson_Uint value)? uint,
    TResult? Function(FfiProbeJson_Float value)? float,
    TResult? Function(FfiProbeJson_Text value)? text,
    TResult? Function(FfiProbeJson_Seq value)? seq,
    TResult? Function(FfiProbeJson_Obj value)? obj,
  }) {
    return float?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiProbeJson_Null value)? null_,
    TResult Function(FfiProbeJson_Bool value)? bool,
    TResult Function(FfiProbeJson_Int value)? int,
    TResult Function(FfiProbeJson_Uint value)? uint,
    TResult Function(FfiProbeJson_Float value)? float,
    TResult Function(FfiProbeJson_Text value)? text,
    TResult Function(FfiProbeJson_Seq value)? seq,
    TResult Function(FfiProbeJson_Obj value)? obj,
    required TResult orElse(),
  }) {
    if (float != null) {
      return float(this);
    }
    return orElse();
  }
}

abstract class FfiProbeJson_Float extends FfiProbeJson {
  const factory FfiProbeJson_Float({required final double value}) =
      _$FfiProbeJson_FloatImpl;
  const FfiProbeJson_Float._() : super._();

  double get value;

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiProbeJson_FloatImplCopyWith<_$FfiProbeJson_FloatImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiProbeJson_TextImplCopyWith<$Res> {
  factory _$$FfiProbeJson_TextImplCopyWith(_$FfiProbeJson_TextImpl value,
          $Res Function(_$FfiProbeJson_TextImpl) then) =
      __$$FfiProbeJson_TextImplCopyWithImpl<$Res>;
  @useResult
  $Res call({String value});
}

/// @nodoc
class __$$FfiProbeJson_TextImplCopyWithImpl<$Res>
    extends _$FfiProbeJsonCopyWithImpl<$Res, _$FfiProbeJson_TextImpl>
    implements _$$FfiProbeJson_TextImplCopyWith<$Res> {
  __$$FfiProbeJson_TextImplCopyWithImpl(_$FfiProbeJson_TextImpl _value,
      $Res Function(_$FfiProbeJson_TextImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? value = null,
  }) {
    return _then(_$FfiProbeJson_TextImpl(
      value: null == value
          ? _value.value
          : value // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class _$FfiProbeJson_TextImpl extends FfiProbeJson_Text {
  const _$FfiProbeJson_TextImpl({required this.value}) : super._();

  @override
  final String value;

  @override
  String toString() {
    return 'FfiProbeJson.text(value: $value)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiProbeJson_TextImpl &&
            (identical(other.value, value) || other.value == value));
  }

  @override
  int get hashCode => Object.hash(runtimeType, value);

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiProbeJson_TextImplCopyWith<_$FfiProbeJson_TextImpl> get copyWith =>
      __$$FfiProbeJson_TextImplCopyWithImpl<_$FfiProbeJson_TextImpl>(
          this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function() null_,
    required TResult Function(bool value) bool,
    required TResult Function(int value) int,
    required TResult Function(BigInt value) uint,
    required TResult Function(double value) float,
    required TResult Function(String value) text,
    required TResult Function(List<FfiProbeJson> items) seq,
    required TResult Function(List<FfiProbeJsonEntry> entries) obj,
  }) {
    return text(value);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function()? null_,
    TResult? Function(bool value)? bool,
    TResult? Function(int value)? int,
    TResult? Function(BigInt value)? uint,
    TResult? Function(double value)? float,
    TResult? Function(String value)? text,
    TResult? Function(List<FfiProbeJson> items)? seq,
    TResult? Function(List<FfiProbeJsonEntry> entries)? obj,
  }) {
    return text?.call(value);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function()? null_,
    TResult Function(bool value)? bool,
    TResult Function(int value)? int,
    TResult Function(BigInt value)? uint,
    TResult Function(double value)? float,
    TResult Function(String value)? text,
    TResult Function(List<FfiProbeJson> items)? seq,
    TResult Function(List<FfiProbeJsonEntry> entries)? obj,
    required TResult orElse(),
  }) {
    if (text != null) {
      return text(value);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiProbeJson_Null value) null_,
    required TResult Function(FfiProbeJson_Bool value) bool,
    required TResult Function(FfiProbeJson_Int value) int,
    required TResult Function(FfiProbeJson_Uint value) uint,
    required TResult Function(FfiProbeJson_Float value) float,
    required TResult Function(FfiProbeJson_Text value) text,
    required TResult Function(FfiProbeJson_Seq value) seq,
    required TResult Function(FfiProbeJson_Obj value) obj,
  }) {
    return text(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiProbeJson_Null value)? null_,
    TResult? Function(FfiProbeJson_Bool value)? bool,
    TResult? Function(FfiProbeJson_Int value)? int,
    TResult? Function(FfiProbeJson_Uint value)? uint,
    TResult? Function(FfiProbeJson_Float value)? float,
    TResult? Function(FfiProbeJson_Text value)? text,
    TResult? Function(FfiProbeJson_Seq value)? seq,
    TResult? Function(FfiProbeJson_Obj value)? obj,
  }) {
    return text?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiProbeJson_Null value)? null_,
    TResult Function(FfiProbeJson_Bool value)? bool,
    TResult Function(FfiProbeJson_Int value)? int,
    TResult Function(FfiProbeJson_Uint value)? uint,
    TResult Function(FfiProbeJson_Float value)? float,
    TResult Function(FfiProbeJson_Text value)? text,
    TResult Function(FfiProbeJson_Seq value)? seq,
    TResult Function(FfiProbeJson_Obj value)? obj,
    required TResult orElse(),
  }) {
    if (text != null) {
      return text(this);
    }
    return orElse();
  }
}

abstract class FfiProbeJson_Text extends FfiProbeJson {
  const factory FfiProbeJson_Text({required final String value}) =
      _$FfiProbeJson_TextImpl;
  const FfiProbeJson_Text._() : super._();

  String get value;

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiProbeJson_TextImplCopyWith<_$FfiProbeJson_TextImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiProbeJson_SeqImplCopyWith<$Res> {
  factory _$$FfiProbeJson_SeqImplCopyWith(_$FfiProbeJson_SeqImpl value,
          $Res Function(_$FfiProbeJson_SeqImpl) then) =
      __$$FfiProbeJson_SeqImplCopyWithImpl<$Res>;
  @useResult
  $Res call({List<FfiProbeJson> items});
}

/// @nodoc
class __$$FfiProbeJson_SeqImplCopyWithImpl<$Res>
    extends _$FfiProbeJsonCopyWithImpl<$Res, _$FfiProbeJson_SeqImpl>
    implements _$$FfiProbeJson_SeqImplCopyWith<$Res> {
  __$$FfiProbeJson_SeqImplCopyWithImpl(_$FfiProbeJson_SeqImpl _value,
      $Res Function(_$FfiProbeJson_SeqImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? items = null,
  }) {
    return _then(_$FfiProbeJson_SeqImpl(
      items: null == items
          ? _value._items
          : items // ignore: cast_nullable_to_non_nullable
              as List<FfiProbeJson>,
    ));
  }
}

/// @nodoc

class _$FfiProbeJson_SeqImpl extends FfiProbeJson_Seq {
  const _$FfiProbeJson_SeqImpl({required final List<FfiProbeJson> items})
      : _items = items,
        super._();

  final List<FfiProbeJson> _items;
  @override
  List<FfiProbeJson> get items {
    if (_items is EqualUnmodifiableListView) return _items;
    // ignore: implicit_dynamic_type
    return EqualUnmodifiableListView(_items);
  }

  @override
  String toString() {
    return 'FfiProbeJson.seq(items: $items)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiProbeJson_SeqImpl &&
            const DeepCollectionEquality().equals(other._items, _items));
  }

  @override
  int get hashCode =>
      Object.hash(runtimeType, const DeepCollectionEquality().hash(_items));

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiProbeJson_SeqImplCopyWith<_$FfiProbeJson_SeqImpl> get copyWith =>
      __$$FfiProbeJson_SeqImplCopyWithImpl<_$FfiProbeJson_SeqImpl>(
          this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function() null_,
    required TResult Function(bool value) bool,
    required TResult Function(int value) int,
    required TResult Function(BigInt value) uint,
    required TResult Function(double value) float,
    required TResult Function(String value) text,
    required TResult Function(List<FfiProbeJson> items) seq,
    required TResult Function(List<FfiProbeJsonEntry> entries) obj,
  }) {
    return seq(items);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function()? null_,
    TResult? Function(bool value)? bool,
    TResult? Function(int value)? int,
    TResult? Function(BigInt value)? uint,
    TResult? Function(double value)? float,
    TResult? Function(String value)? text,
    TResult? Function(List<FfiProbeJson> items)? seq,
    TResult? Function(List<FfiProbeJsonEntry> entries)? obj,
  }) {
    return seq?.call(items);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function()? null_,
    TResult Function(bool value)? bool,
    TResult Function(int value)? int,
    TResult Function(BigInt value)? uint,
    TResult Function(double value)? float,
    TResult Function(String value)? text,
    TResult Function(List<FfiProbeJson> items)? seq,
    TResult Function(List<FfiProbeJsonEntry> entries)? obj,
    required TResult orElse(),
  }) {
    if (seq != null) {
      return seq(items);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiProbeJson_Null value) null_,
    required TResult Function(FfiProbeJson_Bool value) bool,
    required TResult Function(FfiProbeJson_Int value) int,
    required TResult Function(FfiProbeJson_Uint value) uint,
    required TResult Function(FfiProbeJson_Float value) float,
    required TResult Function(FfiProbeJson_Text value) text,
    required TResult Function(FfiProbeJson_Seq value) seq,
    required TResult Function(FfiProbeJson_Obj value) obj,
  }) {
    return seq(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiProbeJson_Null value)? null_,
    TResult? Function(FfiProbeJson_Bool value)? bool,
    TResult? Function(FfiProbeJson_Int value)? int,
    TResult? Function(FfiProbeJson_Uint value)? uint,
    TResult? Function(FfiProbeJson_Float value)? float,
    TResult? Function(FfiProbeJson_Text value)? text,
    TResult? Function(FfiProbeJson_Seq value)? seq,
    TResult? Function(FfiProbeJson_Obj value)? obj,
  }) {
    return seq?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiProbeJson_Null value)? null_,
    TResult Function(FfiProbeJson_Bool value)? bool,
    TResult Function(FfiProbeJson_Int value)? int,
    TResult Function(FfiProbeJson_Uint value)? uint,
    TResult Function(FfiProbeJson_Float value)? float,
    TResult Function(FfiProbeJson_Text value)? text,
    TResult Function(FfiProbeJson_Seq value)? seq,
    TResult Function(FfiProbeJson_Obj value)? obj,
    required TResult orElse(),
  }) {
    if (seq != null) {
      return seq(this);
    }
    return orElse();
  }
}

abstract class FfiProbeJson_Seq extends FfiProbeJson {
  const factory FfiProbeJson_Seq({required final List<FfiProbeJson> items}) =
      _$FfiProbeJson_SeqImpl;
  const FfiProbeJson_Seq._() : super._();

  List<FfiProbeJson> get items;

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiProbeJson_SeqImplCopyWith<_$FfiProbeJson_SeqImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class _$$FfiProbeJson_ObjImplCopyWith<$Res> {
  factory _$$FfiProbeJson_ObjImplCopyWith(_$FfiProbeJson_ObjImpl value,
          $Res Function(_$FfiProbeJson_ObjImpl) then) =
      __$$FfiProbeJson_ObjImplCopyWithImpl<$Res>;
  @useResult
  $Res call({List<FfiProbeJsonEntry> entries});
}

/// @nodoc
class __$$FfiProbeJson_ObjImplCopyWithImpl<$Res>
    extends _$FfiProbeJsonCopyWithImpl<$Res, _$FfiProbeJson_ObjImpl>
    implements _$$FfiProbeJson_ObjImplCopyWith<$Res> {
  __$$FfiProbeJson_ObjImplCopyWithImpl(_$FfiProbeJson_ObjImpl _value,
      $Res Function(_$FfiProbeJson_ObjImpl) _then)
      : super(_value, _then);

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? entries = null,
  }) {
    return _then(_$FfiProbeJson_ObjImpl(
      entries: null == entries
          ? _value._entries
          : entries // ignore: cast_nullable_to_non_nullable
              as List<FfiProbeJsonEntry>,
    ));
  }
}

/// @nodoc

class _$FfiProbeJson_ObjImpl extends FfiProbeJson_Obj {
  const _$FfiProbeJson_ObjImpl({required final List<FfiProbeJsonEntry> entries})
      : _entries = entries,
        super._();

  final List<FfiProbeJsonEntry> _entries;
  @override
  List<FfiProbeJsonEntry> get entries {
    if (_entries is EqualUnmodifiableListView) return _entries;
    // ignore: implicit_dynamic_type
    return EqualUnmodifiableListView(_entries);
  }

  @override
  String toString() {
    return 'FfiProbeJson.obj(entries: $entries)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$FfiProbeJson_ObjImpl &&
            const DeepCollectionEquality().equals(other._entries, _entries));
  }

  @override
  int get hashCode =>
      Object.hash(runtimeType, const DeepCollectionEquality().hash(_entries));

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$FfiProbeJson_ObjImplCopyWith<_$FfiProbeJson_ObjImpl> get copyWith =>
      __$$FfiProbeJson_ObjImplCopyWithImpl<_$FfiProbeJson_ObjImpl>(
          this, _$identity);

  @override
  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function() null_,
    required TResult Function(bool value) bool,
    required TResult Function(int value) int,
    required TResult Function(BigInt value) uint,
    required TResult Function(double value) float,
    required TResult Function(String value) text,
    required TResult Function(List<FfiProbeJson> items) seq,
    required TResult Function(List<FfiProbeJsonEntry> entries) obj,
  }) {
    return obj(entries);
  }

  @override
  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function()? null_,
    TResult? Function(bool value)? bool,
    TResult? Function(int value)? int,
    TResult? Function(BigInt value)? uint,
    TResult? Function(double value)? float,
    TResult? Function(String value)? text,
    TResult? Function(List<FfiProbeJson> items)? seq,
    TResult? Function(List<FfiProbeJsonEntry> entries)? obj,
  }) {
    return obj?.call(entries);
  }

  @override
  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function()? null_,
    TResult Function(bool value)? bool,
    TResult Function(int value)? int,
    TResult Function(BigInt value)? uint,
    TResult Function(double value)? float,
    TResult Function(String value)? text,
    TResult Function(List<FfiProbeJson> items)? seq,
    TResult Function(List<FfiProbeJsonEntry> entries)? obj,
    required TResult orElse(),
  }) {
    if (obj != null) {
      return obj(entries);
    }
    return orElse();
  }

  @override
  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(FfiProbeJson_Null value) null_,
    required TResult Function(FfiProbeJson_Bool value) bool,
    required TResult Function(FfiProbeJson_Int value) int,
    required TResult Function(FfiProbeJson_Uint value) uint,
    required TResult Function(FfiProbeJson_Float value) float,
    required TResult Function(FfiProbeJson_Text value) text,
    required TResult Function(FfiProbeJson_Seq value) seq,
    required TResult Function(FfiProbeJson_Obj value) obj,
  }) {
    return obj(this);
  }

  @override
  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(FfiProbeJson_Null value)? null_,
    TResult? Function(FfiProbeJson_Bool value)? bool,
    TResult? Function(FfiProbeJson_Int value)? int,
    TResult? Function(FfiProbeJson_Uint value)? uint,
    TResult? Function(FfiProbeJson_Float value)? float,
    TResult? Function(FfiProbeJson_Text value)? text,
    TResult? Function(FfiProbeJson_Seq value)? seq,
    TResult? Function(FfiProbeJson_Obj value)? obj,
  }) {
    return obj?.call(this);
  }

  @override
  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(FfiProbeJson_Null value)? null_,
    TResult Function(FfiProbeJson_Bool value)? bool,
    TResult Function(FfiProbeJson_Int value)? int,
    TResult Function(FfiProbeJson_Uint value)? uint,
    TResult Function(FfiProbeJson_Float value)? float,
    TResult Function(FfiProbeJson_Text value)? text,
    TResult Function(FfiProbeJson_Seq value)? seq,
    TResult Function(FfiProbeJson_Obj value)? obj,
    required TResult orElse(),
  }) {
    if (obj != null) {
      return obj(this);
    }
    return orElse();
  }
}

abstract class FfiProbeJson_Obj extends FfiProbeJson {
  const factory FfiProbeJson_Obj(
          {required final List<FfiProbeJsonEntry> entries}) =
      _$FfiProbeJson_ObjImpl;
  const FfiProbeJson_Obj._() : super._();

  List<FfiProbeJsonEntry> get entries;

  /// Create a copy of FfiProbeJson
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$FfiProbeJson_ObjImplCopyWith<_$FfiProbeJson_ObjImpl> get copyWith =>
      throw _privateConstructorUsedError;
}
